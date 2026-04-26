"""
Sortable unique ID generation using ULID.
"""
import ulid as _ulid


def prefixed_id(prefix: str) -> str:
    """Return a prefixed ULID string: '<prefix>_<ulid>'."""
    return f"{prefix}_{_ulid.ULID()}"
