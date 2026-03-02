"""CodeMunch Pro MCP Server — 13 tools for intelligent code indexing.

Supports both stdio and streamable-http transports.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import pathspec

from mcp.server.fastmcp import FastMCP

from codemunch_pro.embedder.embed import Embedder
from codemunch_pro.parser.extractor import extract_symbols
from codemunch_pro.parser.languages import get_language_for_file, LANGUAGES
from codemunch_pro.security import (
    DEFAULT_IGNORE_PATTERNS,
    is_binary_file,
    is_too_large,
    validate_path,
)
from codemunch_pro.storage.database import Database

logger = logging.getLogger(__name__)

# Global state
_databases: dict[str, Database] = {}
_embedder: Embedder | None = None


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def _get_db(repo_path: str) -> Database:
    """Get or create a Database for a repo path."""
    resolved = str(Path(repo_path).resolve())
    if resolved not in _databases:
        _databases[resolved] = Database(resolved)
    return _databases[resolved]


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def _load_gitignore(root: Path) -> pathspec.PathSpec | None:
    """Load .gitignore patterns from a directory."""
    gitignore = root / '.gitignore'
    if gitignore.is_file():
        patterns = gitignore.read_text(errors='replace').splitlines()
        return pathspec.PathSpec.from_lines('gitignore', patterns)
    return None


def _walk_source_files(
    root: Path,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> list[Path]:
    """Walk directory tree and return source files to index."""
    files: list[Path] = []

    # Build ignore spec
    ignore_lines = list(DEFAULT_IGNORE_PATTERNS)
    if exclude_patterns:
        ignore_lines.extend(exclude_patterns)
    ignore_spec = pathspec.PathSpec.from_lines('gitignore', ignore_lines)

    # Load .gitignore
    gitignore_spec = _load_gitignore(root)

    # Build include spec
    include_spec = None
    if include_patterns:
        include_spec = pathspec.PathSpec.from_lines(
            'gitignore', include_patterns,
        )

    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)

        # Filter directories
        dirnames[:] = [
            d for d in dirnames
            if not ignore_spec.match_file(os.path.join(rel_dir, d) + '/')
            and not (gitignore_spec and gitignore_spec.match_file(os.path.join(rel_dir, d) + '/'))
            and not d.startswith('.')
        ]

        for fname in filenames:
            rel_path = os.path.join(rel_dir, fname)
            full_path = Path(dirpath) / fname

            # Skip ignored files
            if ignore_spec.match_file(rel_path):
                continue
            if gitignore_spec and gitignore_spec.match_file(rel_path):
                continue

            # Apply include filter
            if include_spec and not include_spec.match_file(rel_path):
                continue

            # Must be a recognized language
            if get_language_for_file(full_path) is None:
                continue

            # Skip binary/large files
            if is_binary_file(full_path) or is_too_large(full_path):
                continue

            files.append(full_path)

    return files


def _index_directory(
    repo_path: str,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    embed: bool = True,
) -> dict[str, Any]:
    """Core indexing logic used by index_folder and index_repo tools."""
    root = Path(repo_path).resolve()
    if not root.is_dir():
        return {'error': f'Not a directory: {repo_path}'}

    db = _get_db(str(root))
    existing_hashes = db.get_all_file_hashes()
    source_files = _walk_source_files(root, include_patterns, exclude_patterns)

    stats = {
        'total_files': len(source_files),
        'indexed': 0,
        'skipped_unchanged': 0,
        'deleted': 0,
        'symbols_extracted': 0,
        'errors': 0,
    }

    # Track current file paths
    current_paths: set[str] = set()

    # Batch embeddings
    embed_queue: list[tuple[int, str]] = []  # (symbol_id, text)

    for file_path in source_files:
        rel_path = str(file_path.relative_to(root))
        current_paths.add(rel_path)

        # Incremental: skip unchanged files
        file_hash = _sha256_file(file_path)
        if existing_hashes.get(rel_path) == file_hash:
            stats['skipped_unchanged'] += 1
            continue

        try:
            symbols = extract_symbols(file_path)
            language = get_language_for_file(file_path)
            content = file_path.read_text(errors='replace')
            size = file_path.stat().st_size

            file_id = db.upsert_file(
                path=rel_path,
                sha256=file_hash,
                language=language,
                size_bytes=size,
                symbols=symbols,
                file_content=content,
            )

            stats['indexed'] += 1
            stats['symbols_extracted'] += len(symbols)

            # Queue symbols for embedding
            if embed:
                embedder = _get_embedder()
                for sym in symbols:
                    text = embedder.format_symbol_text(
                        name=sym.name,
                        kind=sym.kind,
                        signature=sym.signature,
                        docstring=sym.docstring,
                        language=sym.language,
                    )
                    # We need the symbol ID from the DB
                    sym_row = db.get_symbol(sym.qualified_name)
                    if sym_row:
                        embed_queue.append((sym_row['id'], text))

        except Exception as e:
            logger.warning('Error indexing %s: %s', file_path, e)
            stats['errors'] += 1

    # Remove deleted files
    for old_path in existing_hashes:
        if old_path not in current_paths:
            db.delete_file(old_path)
            stats['deleted'] += 1

    # Resolve call graph edges
    resolved = db.resolve_call_edges()
    stats['call_edges_resolved'] = resolved

    # Batch embed
    if embed and embed_queue:
        try:
            embedder = _get_embedder()
            texts = [text for _, text in embed_queue]
            vectors = embedder.embed(texts)
            items = [
                (sym_id, vec)
                for (sym_id, _), vec in zip(embed_queue, vectors)
            ]
            db.store_embeddings_batch(items)
            stats['embeddings_stored'] = len(items)
        except Exception as e:
            logger.warning('Embedding error: %s', e)
            stats['embedding_error'] = str(e)

    return stats


def create_server(
    transport: str = 'stdio',
    port: int = 5002,
) -> FastMCP:
    """Create and configure the MCP server."""

    kwargs: dict[str, Any] = {
        'name': 'CodeMunch Pro',
    }
    if transport == 'streamable-http':
        kwargs.update({
            'host': '0.0.0.0',
            'port': port,
            'streamable_http_path': '/mcp',
            'stateless_http': True,
            'json_response': True,
        })

    mcp = FastMCP(**kwargs)

    # --- Tool 1: index_folder ---
    @mcp.tool()
    def index_folder(
        path: str,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        embed: bool = True,
    ) -> dict[str, Any]:
        """Index a local directory. Extracts symbols from source files using tree-sitter AST parsing.

        Supports 10 languages: Python, JavaScript, TypeScript, Go, Rust, Java, C, C++, C#, Ruby.
        Uses incremental indexing — only re-parses files that changed (SHA-256 comparison).
        Generates vector embeddings for semantic search.

        Args:
            path: Absolute path to the directory to index.
            include_patterns: Optional glob patterns to include (e.g. ["src/**/*.py"]).
            exclude_patterns: Optional glob patterns to exclude (e.g. ["tests/**"]).
            embed: Whether to generate vector embeddings (default True, set False for faster indexing).
        """
        try:
            validated = validate_path(path)
            if not validated.is_dir():
                return {'error': f'Not a directory: {path}'}
        except ValueError as e:
            return {'error': str(e)}

        return _index_directory(
            str(validated), include_patterns, exclude_patterns, embed,
        )

    # --- Tool 2: index_repo ---
    @mcp.tool()
    def index_repo(
        url: str,
        branch: str = '',
        token: str = '',
        sparse_paths: list[str] | None = None,
        embed: bool = True,
    ) -> dict[str, Any]:
        """Index a GitHub or GitLab repository by downloading its tarball.

        No git binary needed. Downloads the repo archive, extracts, and indexes.
        Subsequent calls use a cache — only re-downloads if new commits exist.

        Args:
            url: Repository URL (e.g. "https://github.com/owner/repo").
            branch: Branch to index (default: repo's default branch).
            token: Optional auth token for private repos (GitHub PAT or GitLab token).
            sparse_paths: Optional list of paths to index (e.g. ["src/", "lib/"]).
            embed: Whether to generate vector embeddings (default True).
        """
        from codemunch_pro.remote import fetch_repo

        try:
            fetch_result = fetch_repo(
                url=url,
                branch=branch,
                token=token,
                sparse_paths=sparse_paths,
            )
        except Exception as e:
            return {'error': f'Failed to fetch repo: {e}'}

        if 'error' in fetch_result:
            return fetch_result

        local_path = fetch_result['local_path']

        # Index the downloaded repo
        index_stats = _index_directory(
            local_path,
            embed=embed,
        )

        return {
            **index_stats,
            'repo_url': url,
            'branch': fetch_result['branch'],
            'sha': fetch_result['sha'],
            'cached': fetch_result['cached'],
            'local_path': local_path,
        }

    # --- Tool 3: list_repos ---
    @mcp.tool()
    def list_repos() -> dict[str, Any]:
        """List all indexed repositories with stats.

        Returns repository paths, file counts, symbol counts, and languages.
        """
        repos = []
        for path, db in _databases.items():
            stats = db.get_stats()
            repos.append(stats)

        # Also check for DB files on disk
        from codemunch_pro.storage.database import DEFAULT_DB_DIR
        if DEFAULT_DB_DIR.exists():
            for db_file in DEFAULT_DB_DIR.glob('*.db'):
                # Open and check if not already loaded
                try:
                    conn = __import__('sqlite3').connect(str(db_file))
                    conn.row_factory = __import__('sqlite3').Row
                    row = conn.execute(
                        "SELECT value FROM meta WHERE key = 'repo_path'"
                    ).fetchone()
                    if row and row['value'] not in _databases:
                        db = _get_db(row['value'])
                        repos.append(db.get_stats())
                    conn.close()
                except Exception:
                    pass

        return {'repos': repos, 'count': len(repos)}

    # --- Tool 4: invalidate_cache ---
    @mcp.tool()
    def invalidate_cache(repo_path: str) -> dict[str, Any]:
        """Force re-index a repository by clearing its cache.

        Deletes the stored hashes so all files are re-parsed on next index.

        Args:
            repo_path: Path to the repository to invalidate.
        """
        db = _get_db(repo_path)
        cur = db.conn.cursor()
        cur.execute('DELETE FROM files')
        cur.execute('DELETE FROM symbols')
        cur.execute('DELETE FROM call_edges')
        cur.execute('DELETE FROM symbols_vec')
        cur.execute("DELETE FROM file_content_fts")
        db.conn.commit()
        return {'status': 'cache_cleared', 'repo': repo_path}

    # --- Tool 5: file_tree ---
    @mcp.tool()
    def file_tree(
        repo_path: str,
        path_prefix: str = '',
        depth: int = 3,
    ) -> dict[str, Any]:
        """Get the directory tree of an indexed repository.

        Shows files with their language, size, and symbol count.

        Args:
            repo_path: Path to the indexed repository.
            path_prefix: Optional path prefix to filter (e.g. "src/").
            depth: Maximum directory depth to show (default 3).
        """
        db = _get_db(repo_path)
        files = db.get_file_tree(path_prefix, depth)

        # Build tree structure
        tree: dict[str, Any] = {}
        for f in files:
            parts = f['path'].split('/')
            # Limit depth
            if len(parts) > depth + 1:
                continue
            node = tree
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = {
                'language': f['language'],
                'size': f['size_bytes'],
                'symbols': f['symbol_count'],
            }

        return {
            'tree': tree,
            'total_files': len(files),
            'repo': repo_path,
        }

    # --- Tool 6: file_outline ---
    @mcp.tool()
    def file_outline(
        repo_path: str,
        file_path: str,
    ) -> dict[str, Any]:
        """List all symbols in a single file, ordered by line number.

        Shows functions, classes, methods, types, and interfaces with their signatures.

        Args:
            repo_path: Path to the indexed repository.
            file_path: Relative path to the file within the repo (e.g. "src/main.py").
        """
        db = _get_db(repo_path)
        symbols = db.get_file_symbols(file_path)
        return {
            'file': file_path,
            'symbols': [
                {
                    'name': s['name'],
                    'qualified_name': s['qualified_name'],
                    'kind': s['kind'],
                    'line': s['line'],
                    'end_line': s['end_line'],
                    'signature': s['signature'],
                    'docstring': (s.get('docstring') or '')[:100],
                }
                for s in symbols
            ],
            'count': len(symbols),
        }

    # --- Tool 7: repo_outline ---
    @mcp.tool()
    def repo_outline(
        repo_path: str,
        kind_filter: str = '',
        limit: int = 200,
    ) -> dict[str, Any]:
        """List all symbols in the repository (summary view).

        Returns a compact list of all functions, classes, methods, etc.
        Use kind_filter to show only specific types.

        Args:
            repo_path: Path to the indexed repository.
            kind_filter: Optional filter: "function", "class", "method", "type", "interface".
            limit: Maximum symbols to return (default 200, max 500).
        """
        db = _get_db(repo_path)
        limit = max(1, min(limit, 500))
        symbols = db.get_all_symbols(kind_filter, limit)
        return {
            'symbols': [
                {
                    'qualified_name': s['qualified_name'],
                    'kind': s['kind'],
                    'file': s['file_path'],
                    'line': s['line'],
                    'signature': s['signature'],
                }
                for s in symbols
            ],
            'count': len(symbols),
            'repo': repo_path,
        }

    # --- Tool 8: get_symbol ---
    @mcp.tool()
    def get_symbol(
        repo_path: str,
        qualified_name: str,
    ) -> dict[str, Any]:
        """Get the full source code of a specific symbol using O(1) byte-offset seek.

        Retrieves the exact bytes of a function, class, or method without reading the entire file.
        This is the key token-saving feature — get exactly what you need.

        Args:
            repo_path: Path to the indexed repository.
            qualified_name: Fully qualified name (e.g. "MyClass.my_method" or "my_function").
        """
        db = _get_db(repo_path)
        sym = db.get_symbol(qualified_name)
        if not sym:
            return {'error': f'Symbol not found: {qualified_name}'}

        # O(1) byte seek to extract source
        file_path = Path(repo_path) / sym['file_path']
        try:
            with open(file_path, 'rb') as f:
                f.seek(sym['byte_offset'])
                source = f.read(sym['byte_length']).decode('utf-8', errors='replace')
        except (OSError, ValueError) as e:
            return {'error': f'Cannot read source: {e}', 'symbol': dict(sym)}

        return {
            'qualified_name': sym['qualified_name'],
            'kind': sym['kind'],
            'language': sym['language'],
            'file': sym['file_path'],
            'line': sym['line'],
            'end_line': sym['end_line'],
            'signature': sym['signature'],
            'docstring': sym.get('docstring', ''),
            'source': source,
            'byte_length': sym['byte_length'],
        }

    # --- Tool 9: get_symbols ---
    @mcp.tool()
    def get_symbols(
        repo_path: str,
        qualified_names: list[str],
    ) -> dict[str, Any]:
        """Batch get source code for multiple symbols at once.

        More efficient than calling get_symbol repeatedly. Returns all requested symbols
        with their full source code.

        Args:
            repo_path: Path to the indexed repository.
            qualified_names: List of fully qualified names to retrieve.
        """
        results = []
        for qname in qualified_names[:50]:  # Cap at 50
            result = get_symbol(repo_path=repo_path, qualified_name=qname)
            results.append(result)

        return {
            'symbols': results,
            'count': len(results),
            'requested': len(qualified_names),
        }

    # --- Tool 10: search_symbols ---
    @mcp.tool()
    def search_symbols(
        repo_path: str,
        query: str,
        kind: str = '',
        limit: int = 20,
    ) -> dict[str, Any]:
        """Hybrid search across all symbols using FTS5 + vector similarity.

        Combines keyword matching (BM25) with semantic similarity (embeddings)
        using Reciprocal Rank Fusion for best results.

        Args:
            repo_path: Path to the indexed repository.
            query: Search query (natural language or code terms).
            kind: Optional filter: "function", "class", "method", "type", "interface".
            limit: Maximum results (default 20).
        """
        db = _get_db(repo_path)
        limit = max(1, min(limit, 50))

        # FTS5 search
        fts_results = db.search_fts(query, limit * 2)

        # Vector search
        vec_results = []
        try:
            embedder = _get_embedder()
            query_vec = embedder.embed_one(query)
            if query_vec:
                vec_results = db.search_vec(query_vec, limit * 2)
        except Exception as e:
            logger.warning('Vector search failed: %s', e)

        # Reciprocal Rank Fusion (k=60)
        scores: dict[str, float] = {}
        symbol_data: dict[str, dict] = {}
        K = 60

        for rank, r in enumerate(fts_results):
            qn = r['qualified_name']
            scores[qn] = scores.get(qn, 0) + 1 / (K + rank)
            symbol_data[qn] = r

        for rank, r in enumerate(vec_results):
            qn = r['qualified_name']
            scores[qn] = scores.get(qn, 0) + 1 / (K + rank)
            if qn not in symbol_data:
                symbol_data[qn] = r

        # Sort by fused score
        ranked = sorted(scores.items(), key=lambda x: -x[1])

        # Apply kind filter
        results = []
        for qn, score in ranked:
            if kind and symbol_data[qn].get('kind') != kind:
                continue
            sym = symbol_data[qn]
            results.append({
                'qualified_name': sym['qualified_name'],
                'kind': sym['kind'],
                'file': sym.get('file_path', ''),
                'line': sym['line'],
                'signature': sym.get('signature', ''),
                'docstring': (sym.get('docstring') or '')[:100],
                'score': round(score, 4),
            })
            if len(results) >= limit:
                break

        return {
            'results': results,
            'count': len(results),
            'query': query,
            'search_type': 'hybrid' if vec_results else 'fts_only',
        }

    # --- Tool 11: search_text ---
    @mcp.tool()
    def search_text(
        repo_path: str,
        query: str,
        glob: str = '',
        limit: int = 20,
    ) -> dict[str, Any]:
        """Full-text search in file contents (strings, comments, config values).

        Searches the raw text of all indexed files, not just symbol names.
        Useful for finding string literals, TODO comments, config values, error messages.

        Args:
            repo_path: Path to the indexed repository.
            query: Text to search for.
            glob: Optional file glob pattern (e.g. "*.py", "src/**/*.ts").
            limit: Maximum results (default 20).
        """
        db = _get_db(repo_path)
        limit = max(1, min(limit, 50))
        results = db.search_text(query, glob, limit)
        return {
            'results': results,
            'count': len(results),
            'query': query,
        }

    # --- Tool 12: get_callees ---
    @mcp.tool()
    def get_callees(
        repo_path: str,
        qualified_name: str,
        depth: int = 1,
    ) -> dict[str, Any]:
        """Get what a function calls (outgoing call graph edges).

        Traces the call graph from a function to see what it invokes.
        Use depth > 1 for transitive callees.

        Args:
            repo_path: Path to the indexed repository.
            qualified_name: Fully qualified name of the function to trace from.
            depth: How deep to traverse (1 = direct calls, 2 = calls of calls, etc).
        """
        db = _get_db(repo_path)
        depth = max(1, min(depth, 5))
        callees = db.get_callees(qualified_name, depth)
        return {
            'function': qualified_name,
            'callees': callees,
            'count': len(callees),
            'depth': depth,
        }

    # --- Tool 13: get_callers ---
    @mcp.tool()
    def get_callers(
        repo_path: str,
        qualified_name: str,
        depth: int = 1,
    ) -> dict[str, Any]:
        """Get who calls a function (incoming call graph edges).

        Traces the call graph backwards to find all callers of a function.
        Use depth > 1 for transitive callers.

        Args:
            repo_path: Path to the indexed repository.
            qualified_name: Fully qualified name of the function to trace.
            depth: How deep to traverse (1 = direct callers, 2 = callers of callers, etc).
        """
        db = _get_db(repo_path)
        depth = max(1, min(depth, 5))
        callers = db.get_callers(qualified_name, depth)
        return {
            'function': qualified_name,
            'callers': callers,
            'count': len(callers),
            'depth': depth,
        }

    return mcp
