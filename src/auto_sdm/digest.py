"""Content addressing.

Artifacts are named by the hash of their bytes and steps by the hash of their inputs.
Rewind depends on the latter: a stage re-runs when and only when its inputs digest
changes, so a digest that quietly ignores an input serves a stale result instead.

Unhashable types therefore raise rather than degrading to something like ``repr``, which
would embed a memory address and make every rerun look different — or worse, make two
genuinely different inputs look the same.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

_CHUNK = 1 << 20


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def canonical(value: Any) -> Any:
    """Reduce a value to an order-stable, JSON-encodable form.

    Arrays reduce to shape, dtype, and a hash of their buffer rather than their contents,
    so digesting a large predictor stack stays cheap without losing sensitivity to it.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        # repr round-trips exactly; str() does not for every float.
        return {"__float__": repr(value)}
    if isinstance(value, bytes):
        return {"__bytes__": digest_bytes(value)}
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "__array__": {
                "shape": list(contiguous.shape),
                "dtype": str(contiguous.dtype),
                "buffer": digest_bytes(contiguous.tobytes()),
            }
        }
    if isinstance(value, np.generic):
        return canonical(value.item())
    if isinstance(value, BaseModel):
        return canonical(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        f"cannot digest {type(value).__name__}; add a canonical form rather than letting "
        "it hash unstably"
    )


def digest_obj(value: Any) -> str:
    """Hash any canonicalisable value. Stable across processes and key orderings."""
    encoded = json.dumps(canonical(value), sort_keys=True, separators=(",", ":"))
    return digest_bytes(encoded.encode("utf-8"))
