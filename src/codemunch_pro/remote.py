"""Remote repository fetching — download and cache GitHub/GitLab repos via API.

Downloads repo tarballs (no git binary needed), extracts to cache dir,
supports incremental updates by checking commit SHAs.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.parse import quote_plus

import httpx

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / '.codemunch-pro' / 'repos'
_GITHUB_RE = re.compile(
    r'(?:https?://)?github\.com/([^/]+)/([^/.\s]+?)(?:\.git)?(?:/.*)?$'
)
_GITLAB_RE = re.compile(
    r'(?:https?://)?gitlab\.com/([^/]+(?:/[^/]+)*)/([^/.\s]+?)(?:\.git)?(?:/.*)?$'
)
_GENERIC_GIT_RE = re.compile(
    r'(?:https?://)?([^/]+)/.*?([^/.\s]+?)(?:\.git)?$'
)

USER_AGENT = 'CodeMunch-Pro/1.1 (https://github.com/BigJai/codemunch-pro)'
DOWNLOAD_TIMEOUT = 120.0  # seconds


def parse_repo_url(url: str) -> dict:
    """Parse a repository URL into components.

    Returns:
        dict with keys: host, owner, repo, url_type ('github'|'gitlab'|'generic')
    """
    url = url.strip()

    m = _GITHUB_RE.match(url)
    if m:
        return {
            'host': 'github.com',
            'owner': m.group(1),
            'repo': m.group(2),
            'url_type': 'github',
        }

    m = _GITLAB_RE.match(url)
    if m:
        return {
            'host': 'gitlab.com',
            'owner': m.group(1),
            'repo': m.group(2),
            'url_type': 'gitlab',
        }

    m = _GENERIC_GIT_RE.match(url)
    if m:
        return {
            'host': m.group(1),
            'owner': '',
            'repo': m.group(2),
            'url_type': 'generic',
        }

    raise ValueError(f'Cannot parse repository URL: {url}')


def _cache_dir_for_repo(
    parsed: dict,
    branch: str,
    base_dir: Path | None = None,
) -> Path:
    """Get the cache directory for a repo + branch combo."""
    cache_base = base_dir or DEFAULT_CACHE_DIR
    safe_name = f"{parsed['owner']}_{parsed['repo']}_{branch or 'default'}"
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', safe_name)
    return cache_base / safe_name


def _get_meta_path(cache_dir: Path) -> Path:
    return cache_dir / '.codemunch-meta.json'


def _read_meta(cache_dir: Path) -> dict:
    meta_path = _get_meta_path(cache_dir)
    if meta_path.is_file():
        return json.loads(meta_path.read_text())
    return {}


def _write_meta(cache_dir: Path, meta: dict) -> None:
    meta_path = _get_meta_path(cache_dir)
    meta_path.write_text(json.dumps(meta, indent=2))


def _github_get_default_branch(
    owner: str, repo: str, token: str = '',
) -> str:
    """Get the default branch for a GitHub repo."""
    headers = {'User-Agent': USER_AGENT}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    resp = httpx.get(
        f'https://api.github.com/repos/{owner}/{repo}',
        headers=headers,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()['default_branch']


def _github_get_latest_sha(
    owner: str, repo: str, branch: str, token: str = '',
) -> str:
    """Get the latest commit SHA for a GitHub branch."""
    headers = {'User-Agent': USER_AGENT}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    resp = httpx.get(
        f'https://api.github.com/repos/{owner}/{repo}/commits/{branch}',
        headers=headers,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()['sha']


def _gitlab_get_default_branch(
    owner: str, repo: str, token: str = '',
) -> str:
    """Get the default branch for a GitLab repo."""
    project_id = quote_plus(f'{owner}/{repo}')
    headers = {}
    if token:
        headers['PRIVATE-TOKEN'] = token
    resp = httpx.get(
        f'https://gitlab.com/api/v4/projects/{project_id}',
        headers=headers,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()['default_branch']


def _gitlab_get_latest_sha(
    owner: str, repo: str, branch: str, token: str = '',
) -> str:
    """Get the latest commit SHA for a GitLab branch."""
    project_id = quote_plus(f'{owner}/{repo}')
    headers = {}
    if token:
        headers['PRIVATE-TOKEN'] = token
    resp = httpx.get(
        f'https://gitlab.com/api/v4/projects/{project_id}/repository/branches/{quote_plus(branch)}',
        headers=headers,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()['commit']['id']


def _download_github_tarball(
    owner: str,
    repo: str,
    branch: str,
    dest: Path,
    token: str = '',
) -> None:
    """Download and extract a GitHub repo tarball."""
    headers = {'User-Agent': USER_AGENT}
    if token:
        headers['Authorization'] = f'Bearer {token}'

    url = f'https://api.github.com/repos/{owner}/{repo}/tarball/{branch}'

    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with httpx.stream('GET', url, headers=headers, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as resp:
            resp.raise_for_status()
            with open(tmp_path, 'wb') as f:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    f.write(chunk)

        _safe_extract_tarball(tmp_path, dest)
    finally:
        tmp_path.unlink(missing_ok=True)


def _download_gitlab_tarball(
    owner: str,
    repo: str,
    branch: str,
    dest: Path,
    token: str = '',
    sparse_path: str = '',
) -> None:
    """Download and extract a GitLab repo tarball.

    GitLab supports server-side sparse download via the `path` param.
    """
    project_id = quote_plus(f'{owner}/{repo}')
    headers = {}
    if token:
        headers['PRIVATE-TOKEN'] = token

    url = f'https://gitlab.com/api/v4/projects/{project_id}/repository/archive.tar.gz?sha={quote_plus(branch)}'
    if sparse_path:
        url += f'&path={quote_plus(sparse_path)}'

    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with httpx.stream('GET', url, headers=headers, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as resp:
            resp.raise_for_status()
            with open(tmp_path, 'wb') as f:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    f.write(chunk)

        _safe_extract_tarball(tmp_path, dest)
    finally:
        tmp_path.unlink(missing_ok=True)


def _safe_extract_tarball(tarball_path: Path, dest: Path) -> None:
    """Safely extract a tarball, preventing path traversal and symlink attacks.

    GitHub/GitLab tarballs have a top-level directory like 'owner-repo-sha/'.
    We strip this prefix and extract directly into dest.
    """
    dest.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tarball_path, 'r:gz') as tar:
        # Find the common prefix (top-level dir in the tarball)
        members = tar.getmembers()
        if not members:
            return

        # Detect top-level directory prefix
        prefix = ''
        first = members[0].name
        if '/' in first:
            prefix = first.split('/')[0] + '/'
        elif members[0].isdir():
            prefix = first + '/'

        for member in members:
            # Strip top-level prefix
            if prefix and member.name.startswith(prefix):
                member_path = member.name[len(prefix):]
            else:
                member_path = member.name

            if not member_path:
                continue

            # Security: prevent path traversal
            target = (dest / member_path).resolve()
            if not str(target).startswith(str(dest.resolve())):
                logger.warning('Skipping path traversal attempt: %s', member.name)
                continue

            # Security: skip symlinks
            if member.issym() or member.islnk():
                logger.debug('Skipping symlink: %s', member.name)
                continue

            # Extract
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src:
                    if src:
                        target.write_bytes(src.read())


def fetch_repo(
    url: str,
    branch: str = '',
    token: str = '',
    cache_dir: Path | None = None,
    sparse_paths: list[str] | None = None,
) -> dict:
    """Fetch a remote repository — download tarball, extract, cache.

    Returns:
        dict with keys:
        - local_path: Path to the extracted repo on disk
        - cached: whether we used a cached version (no download needed)
        - branch: resolved branch name
        - sha: latest commit SHA
        - url_type: 'github' or 'gitlab'
    """
    parsed = parse_repo_url(url)
    url_type = parsed['url_type']
    owner = parsed['owner']
    repo = parsed['repo']

    if url_type not in ('github', 'gitlab'):
        return {
            'error': f"Unsupported host: {parsed['host']}. Only GitHub and GitLab are supported.",
            'hint': 'Clone the repo locally and use index_folder instead.',
        }

    # Resolve default branch if not specified
    if not branch:
        if url_type == 'github':
            branch = _github_get_default_branch(owner, repo, token)
        else:
            branch = _gitlab_get_default_branch(owner, repo, token)

    # Check latest commit SHA
    if url_type == 'github':
        latest_sha = _github_get_latest_sha(owner, repo, branch, token)
    else:
        latest_sha = _gitlab_get_latest_sha(owner, repo, branch, token)

    # Get cache directory
    repo_cache = _cache_dir_for_repo(parsed, branch, cache_dir)
    meta = _read_meta(repo_cache)

    # Check if cache is fresh
    if (
        repo_cache.is_dir()
        and meta.get('sha') == latest_sha
        and any(repo_cache.iterdir())
    ):
        logger.info('Cache hit for %s/%s@%s (sha=%s)', owner, repo, branch, latest_sha[:8])
        return {
            'local_path': str(repo_cache),
            'cached': True,
            'branch': branch,
            'sha': latest_sha,
            'url_type': url_type,
            'owner': owner,
            'repo': repo,
        }

    # Cache miss — download fresh
    logger.info('Downloading %s/%s@%s ...', owner, repo, branch)

    # Clean old cache
    if repo_cache.is_dir():
        shutil.rmtree(repo_cache)

    # Download and extract
    if url_type == 'github':
        _download_github_tarball(owner, repo, branch, repo_cache, token)
    else:
        _download_gitlab_tarball(owner, repo, branch, repo_cache, token)

    # Filter to sparse_paths if requested
    if sparse_paths:
        _apply_sparse_filter(repo_cache, sparse_paths)

    # Write meta
    _write_meta(repo_cache, {
        'url': url,
        'owner': owner,
        'repo': repo,
        'branch': branch,
        'sha': latest_sha,
        'url_type': url_type,
    })

    return {
        'local_path': str(repo_cache),
        'cached': False,
        'branch': branch,
        'sha': latest_sha,
        'url_type': url_type,
        'owner': owner,
        'repo': repo,
    }


def _apply_sparse_filter(repo_dir: Path, sparse_paths: list[str]) -> None:
    """Delete files not matching sparse_paths patterns."""
    keep_prefixes = [p.rstrip('/') for p in sparse_paths]

    for item in list(repo_dir.rglob('*')):
        if item.name == '.codemunch-meta.json':
            continue
        if not item.is_file():
            continue

        rel = str(item.relative_to(repo_dir))
        keep = any(
            rel.startswith(prefix) or rel.startswith(prefix + '/')
            for prefix in keep_prefixes
        )
        if not keep:
            item.unlink()

    # Clean empty directories
    for item in sorted(repo_dir.rglob('*'), reverse=True):
        if item.is_dir() and not any(item.iterdir()):
            item.rmdir()
