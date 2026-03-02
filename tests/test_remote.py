"""Tests for remote repo fetching — URL parsing, tarball extraction, caching."""

import json
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codemunch_pro.remote import (
    _apply_sparse_filter,
    _cache_dir_for_repo,
    _safe_extract_tarball,
    fetch_repo,
    parse_repo_url,
)


class TestParseRepoUrl:
    def test_github_https(self):
        r = parse_repo_url('https://github.com/BigJai/codemunch-pro')
        assert r['host'] == 'github.com'
        assert r['owner'] == 'BigJai'
        assert r['repo'] == 'codemunch-pro'
        assert r['url_type'] == 'github'

    def test_github_with_git_suffix(self):
        r = parse_repo_url('https://github.com/BigJai/codemunch-pro.git')
        assert r['owner'] == 'BigJai'
        assert r['repo'] == 'codemunch-pro'
        assert r['url_type'] == 'github'

    def test_github_with_trailing_slash(self):
        r = parse_repo_url('https://github.com/facebook/react/')
        assert r['owner'] == 'facebook'
        assert r['repo'] == 'react'

    def test_github_with_path(self):
        r = parse_repo_url('https://github.com/owner/repo/tree/main/src')
        assert r['owner'] == 'owner'
        assert r['repo'] == 'repo'

    def test_github_no_protocol(self):
        r = parse_repo_url('github.com/BigJai/codemunch-pro')
        assert r['url_type'] == 'github'
        assert r['owner'] == 'BigJai'

    def test_gitlab_https(self):
        r = parse_repo_url('https://gitlab.com/group/project')
        assert r['host'] == 'gitlab.com'
        assert r['owner'] == 'group'
        assert r['repo'] == 'project'
        assert r['url_type'] == 'gitlab'

    def test_gitlab_nested_group(self):
        r = parse_repo_url('https://gitlab.com/org/subgroup/project')
        assert r['owner'] == 'org/subgroup'
        assert r['repo'] == 'project'
        assert r['url_type'] == 'gitlab'

    def test_invalid_url(self):
        with pytest.raises(ValueError, match='Cannot parse'):
            parse_repo_url('not-a-url')

    def test_whitespace_stripped(self):
        r = parse_repo_url('  https://github.com/a/b  ')
        assert r['owner'] == 'a'
        assert r['repo'] == 'b'


class TestCacheDir:
    def test_returns_path(self, tmp_path):
        parsed = {'owner': 'BigJai', 'repo': 'test', 'url_type': 'github'}
        result = _cache_dir_for_repo(parsed, 'main', tmp_path)
        assert result.parent == tmp_path
        assert 'BigJai_test_main' in result.name

    def test_sanitizes_special_chars(self, tmp_path):
        parsed = {'owner': 'org/sub', 'repo': 'my-repo', 'url_type': 'gitlab'}
        result = _cache_dir_for_repo(parsed, 'feat/xyz', tmp_path)
        # Should not contain slashes
        assert '/' not in result.name

    def test_default_branch_label(self, tmp_path):
        parsed = {'owner': 'a', 'repo': 'b', 'url_type': 'github'}
        result = _cache_dir_for_repo(parsed, '', tmp_path)
        assert 'default' in result.name


class TestSafeExtractTarball:
    def _make_tarball(self, tmp_path, files: dict[str, str], prefix: str = 'repo-abc123/') -> Path:
        """Create a test tarball with given files."""
        tar_path = tmp_path / 'test.tar.gz'
        with tarfile.open(tar_path, 'w:gz') as tar:
            for name, content in files.items():
                full_name = prefix + name
                data = content.encode()
                import io
                info = tarfile.TarInfo(name=full_name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        return tar_path

    def test_extracts_files(self, tmp_path):
        tar = self._make_tarball(tmp_path, {
            'src/main.py': 'print("hello")',
            'README.md': '# Test',
        })
        dest = tmp_path / 'extracted'
        _safe_extract_tarball(tar, dest)

        assert (dest / 'src' / 'main.py').read_text() == 'print("hello")'
        assert (dest / 'README.md').read_text() == '# Test'

    def test_strips_prefix(self, tmp_path):
        tar = self._make_tarball(
            tmp_path,
            {'file.txt': 'data'},
            prefix='owner-repo-abcdef1234/',
        )
        dest = tmp_path / 'out'
        _safe_extract_tarball(tar, dest)

        # Should NOT have the prefix directory
        assert not (dest / 'owner-repo-abcdef1234').exists()
        assert (dest / 'file.txt').read_text() == 'data'

    def test_blocks_path_traversal(self, tmp_path):
        """Tarballs with ../escape paths should be skipped."""
        tar_path = tmp_path / 'evil.tar.gz'
        with tarfile.open(tar_path, 'w:gz') as tar:
            import io
            info = tarfile.TarInfo(name='prefix/../../../etc/evil')
            info.size = 4
            tar.addfile(info, io.BytesIO(b'evil'))

        dest = tmp_path / 'safe'
        _safe_extract_tarball(tar_path, dest)

        # The evil file should NOT exist outside dest
        assert not (tmp_path / 'etc' / 'evil').exists()

    def test_empty_tarball(self, tmp_path):
        tar_path = tmp_path / 'empty.tar.gz'
        with tarfile.open(tar_path, 'w:gz'):
            pass
        dest = tmp_path / 'out'
        _safe_extract_tarball(tar_path, dest)
        # Should not crash


class TestSparseFilter:
    def test_keeps_matching_paths(self, tmp_path):
        (tmp_path / 'src').mkdir()
        (tmp_path / 'src' / 'main.py').write_text('code')
        (tmp_path / 'tests').mkdir()
        (tmp_path / 'tests' / 'test.py').write_text('test')
        (tmp_path / 'README.md').write_text('readme')

        _apply_sparse_filter(tmp_path, ['src/'])

        assert (tmp_path / 'src' / 'main.py').exists()
        assert not (tmp_path / 'tests' / 'test.py').exists()
        assert not (tmp_path / 'README.md').exists()

    def test_keeps_multiple_paths(self, tmp_path):
        (tmp_path / 'src').mkdir()
        (tmp_path / 'src' / 'a.py').write_text('a')
        (tmp_path / 'lib').mkdir()
        (tmp_path / 'lib' / 'b.py').write_text('b')
        (tmp_path / 'docs').mkdir()
        (tmp_path / 'docs' / 'c.md').write_text('c')

        _apply_sparse_filter(tmp_path, ['src', 'lib'])

        assert (tmp_path / 'src' / 'a.py').exists()
        assert (tmp_path / 'lib' / 'b.py').exists()
        assert not (tmp_path / 'docs' / 'c.md').exists()


class TestFetchRepoIntegration:
    """Integration tests that actually hit GitHub API (small public repo)."""

    @pytest.mark.network
    def test_fetch_github_repo(self, tmp_path):
        """Fetch a small real GitHub repo."""
        result = fetch_repo(
            'https://github.com/BigJai/codemunch-pro',
            cache_dir=tmp_path,
        )

        assert 'error' not in result
        assert result['url_type'] == 'github'
        assert result['owner'] == 'BigJai'
        assert result['repo'] == 'codemunch-pro'
        assert result['branch']  # Should have resolved default branch
        assert result['sha']  # Should have latest SHA
        assert Path(result['local_path']).is_dir()

        # Should have actual files
        local = Path(result['local_path'])
        assert any(local.rglob('*.py'))

    @pytest.mark.network
    def test_cache_hit(self, tmp_path):
        """Second fetch should use cache."""
        result1 = fetch_repo(
            'https://github.com/BigJai/codemunch-pro',
            cache_dir=tmp_path,
        )
        assert not result1.get('cached')

        result2 = fetch_repo(
            'https://github.com/BigJai/codemunch-pro',
            cache_dir=tmp_path,
        )
        assert result2.get('cached')
        assert result2['local_path'] == result1['local_path']

    def test_unsupported_host(self, tmp_path):
        result = fetch_repo(
            'https://bitbucket.org/owner/repo',
            cache_dir=tmp_path,
        )
        assert 'error' in result

    @pytest.mark.network
    def test_fetch_with_branch(self, tmp_path):
        """Fetch a specific branch."""
        result = fetch_repo(
            'https://github.com/BigJai/codemunch-pro',
            branch='main',
            cache_dir=tmp_path,
        )
        assert result['branch'] == 'main'

    @pytest.mark.network
    def test_fetch_with_sparse_paths(self, tmp_path):
        """Sparse checkout should only keep specified paths."""
        result = fetch_repo(
            'https://github.com/BigJai/codemunch-pro',
            sparse_paths=['src/'],
            cache_dir=tmp_path,
        )
        local = Path(result['local_path'])
        # Should have src/ files
        assert (local / 'src').is_dir()
        # Should NOT have tests/ or top-level files
        assert not (local / 'tests').exists()
