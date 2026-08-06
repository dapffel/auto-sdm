"""The artifact store.

Artifacts are content-addressed, so writing the same array at five stages of a candidate
costs one file and five references. That is not only a space win: it makes "this stage
did not change the data" a checkable fact rather than a claim in a step card.

Run records live under a per-run directory and are plain JSON, because the thing reading
them next may be a report generator, a web app, or a later run mining history.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .digest import digest_bytes
from .models import Artifact, RunManifest, RunResult, StepRecord


class ArtifactStore:
    """Content-addressed blobs plus per-run records, rooted at one directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.blobs = self.root / "artifacts"
        self.runs = self.root / "runs"
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.runs.mkdir(parents=True, exist_ok=True)

    # -- blobs ---------------------------------------------------------------------

    def put_bytes(self, name: str, data: bytes, media_type: str, suffix: str = "") -> Artifact:
        digest = digest_bytes(data)
        path = self.blobs / f"{digest}{suffix}"
        if not path.exists():
            # Write-once: identical content is already identical bytes on disk.
            path.write_bytes(data)
        return Artifact(
            name=name,
            path=str(path.relative_to(self.root)),
            media_type=media_type,
            digest=digest,
        )

    def put_json(self, name: str, payload: object) -> Artifact:
        encoded = json.dumps(payload, sort_keys=True, indent=2).encode("utf-8")
        return self.put_bytes(name, encoded, "application/json", ".json")

    def put_array(self, name: str, array: np.ndarray) -> Artifact:
        contiguous = np.ascontiguousarray(array)
        header = f"{contiguous.dtype}|{contiguous.shape}".encode("utf-8")
        return self.put_bytes(
            name,
            header + b"\0" + contiguous.tobytes(),
            "application/x-auto-sdm-array",
            ".arr",
        )

    def read_bytes(self, artifact: Artifact) -> bytes:
        return (self.root / artifact.path).read_bytes()

    # -- run records ---------------------------------------------------------------

    def run_dir(self, run_id: str) -> Path:
        path = self.runs / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_manifest(self, manifest: RunManifest) -> Path:
        path = self.run_dir(manifest.run_id) / "manifest.json"
        path.write_text(manifest.model_dump_json(indent=2))
        return path

    def write_step_record(self, run_id: str, index: int, record: StepRecord) -> Path:
        steps = self.run_dir(run_id) / "candidates" / record.candidate_id / "steps"
        steps.mkdir(parents=True, exist_ok=True)
        path = steps / f"{index:02d}-{record.stage}.json"
        path.write_text(record.model_dump_json(indent=2))
        return path

    def write_result(self, result: RunResult) -> Path:
        path = self.run_dir(result.manifest.run_id) / "result.json"
        path.write_text(result.model_dump_json(indent=2))
        return path
