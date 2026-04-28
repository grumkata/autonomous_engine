"""
orchestrator/tools/file_store.py — Per-project disk file storage.

Every project gets a folder: ./workspace_files/{project_id}/
Files are organised by type:
    images/     — .png, .jpg, .svg generated images and diagrams
    documents/  — .pdf, .docx, .txt output documents
    data/       — .xlsx, .csv, .json datasets
    audio/      — .mp3, .wav generated audio
    code/       — .py, .js, .html generated code files
    raw/        — anything else

File metadata is tracked in memory per session (not persisted to DB —
the files themselves are the ground truth).  The workspace_manager stores
artifact_ids which are just file paths relative to the workspace root.
"""

from __future__ import annotations

import hashlib
import mimetypes
import shutil
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

# Root folder next to the project — survives restarts, not gitignored by default
_WORKSPACE_ROOT = Path("./workspace_files")


class ResourceType(StrEnum):
    IMAGE     = "image"
    DIAGRAM   = "diagram"
    DOCUMENT  = "document"
    DATA      = "data"
    AUDIO     = "audio"
    CODE      = "code"
    WEBPAGE   = "webpage"
    RAW       = "raw"


_TYPE_DIRS: dict[ResourceType, str] = {
    ResourceType.IMAGE:    "images",
    ResourceType.DIAGRAM:  "images",
    ResourceType.DOCUMENT: "documents",
    ResourceType.DATA:     "data",
    ResourceType.AUDIO:    "audio",
    ResourceType.CODE:     "code",
    ResourceType.WEBPAGE:  "webpages",
    ResourceType.RAW:      "raw",
}

_EXT_TO_TYPE: dict[str, ResourceType] = {
    ".png": ResourceType.IMAGE, ".jpg": ResourceType.IMAGE,
    ".jpeg": ResourceType.IMAGE, ".gif": ResourceType.IMAGE,
    ".webp": ResourceType.IMAGE, ".svg": ResourceType.DIAGRAM,
    ".pdf": ResourceType.DOCUMENT, ".docx": ResourceType.DOCUMENT,
    ".txt": ResourceType.DOCUMENT, ".md": ResourceType.DOCUMENT,
    ".xlsx": ResourceType.DATA, ".csv": ResourceType.DATA,
    ".json": ResourceType.DATA,
    ".mp3": ResourceType.AUDIO, ".wav": ResourceType.AUDIO,
    ".ogg": ResourceType.AUDIO,
    ".py": ResourceType.CODE, ".js": ResourceType.CODE,
    ".ts": ResourceType.CODE, ".html": ResourceType.CODE,
    ".htm": ResourceType.WEBPAGE,
}


def project_dir(project_id: str) -> Path:
    d = _WORKSPACE_ROOT / project_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def type_dir(project_id: str, resource_type: ResourceType) -> Path:
    d = project_dir(project_id) / _TYPE_DIRS[resource_type]
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_bytes(
    project_id: str,
    filename: str,
    data: bytes,
    resource_type: ResourceType | None = None,
) -> Path:
    """
    Save raw bytes to the project workspace.
    Returns the full Path where the file was written.
    Auto-detects resource_type from extension if not provided.
    """
    ext = Path(filename).suffix.lower()
    rtype = resource_type or _EXT_TO_TYPE.get(ext, ResourceType.RAW)
    dest = type_dir(project_id, rtype) / filename
    dest.write_bytes(data)
    log.info("file_store.saved", project_id=project_id, path=str(dest), bytes=len(data))
    return dest


def save_text(
    project_id: str,
    filename: str,
    text: str,
    resource_type: ResourceType | None = None,
    encoding: str = "utf-8",
) -> Path:
    """Save text content to the project workspace."""
    return save_bytes(project_id, filename, text.encode(encoding), resource_type)


def read_bytes(project_id: str, relative_path: str) -> bytes:
    """Read a file from the project workspace by relative path."""
    full = project_dir(project_id) / relative_path
    return full.read_bytes()


def list_files(project_id: str, resource_type: ResourceType | None = None) -> list[dict]:
    """
    List files in the project workspace.
    Returns dicts with: path, name, type, size_bytes, created_at
    """
    base = project_dir(project_id)
    results = []

    if resource_type:
        dirs = [type_dir(project_id, resource_type)]
    else:
        dirs = [base / d for d in _TYPE_DIRS.values() if (base / d).exists()]

    seen: set[str] = set()
    for d in dirs:
        for f in sorted(d.rglob("*")):
            if f.is_file() and str(f) not in seen:
                seen.add(str(f))
                rel = str(f.relative_to(base))
                ext = f.suffix.lower()
                results.append({
                    "path": rel,
                    "name": f.name,
                    "type": _EXT_TO_TYPE.get(ext, ResourceType.RAW).value,
                    "size_bytes": f.stat().st_size,
                    "full_path": str(f),
                    "url_path": f"/projects/{project_id}/files/{rel}",
                })
    return results


def delete_project_files(project_id: str) -> int:
    """Remove all files for a project. Returns count deleted."""
    d = _WORKSPACE_ROOT / project_id
    if d.exists():
        count = sum(1 for f in d.rglob("*") if f.is_file())
        shutil.rmtree(d)
        log.info("file_store.deleted_project", project_id=project_id, files=count)
        return count
    return 0


def unique_filename(base: str, project_id: str, resource_type: ResourceType) -> str:
    """Generate a filename that won't collide with existing files."""
    stem = Path(base).stem
    ext  = Path(base).suffix or ".bin"
    dest_dir = type_dir(project_id, resource_type)
    candidate = f"{stem}{ext}"
    counter = 1
    while (dest_dir / candidate).exists():
        candidate = f"{stem}_{counter}{ext}"
        counter += 1
    return candidate
