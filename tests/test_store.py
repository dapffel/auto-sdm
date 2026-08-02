from __future__ import annotations

import numpy as np

from auto_sdm import Artifact, ArtifactStore, RunManifest, StepRecord


def test_identical_content_is_stored_once(store: ArtifactStore) -> None:
    """Content addressing is what makes 'this stage changed nothing' checkable."""
    first = store.put_bytes("a", b"same", "text/plain")
    second = store.put_bytes("b", b"same", "text/plain")
    assert first.digest == second.digest
    assert first.path == second.path
    assert len(list(store.blobs.iterdir())) == 1


def test_different_content_gets_different_paths(store: ArtifactStore) -> None:
    first = store.put_bytes("a", b"one", "text/plain")
    second = store.put_bytes("a", b"two", "text/plain")
    assert first.path != second.path
    assert len(list(store.blobs.iterdir())) == 2


def test_arrays_round_trip_through_the_store(store: ArtifactStore) -> None:
    array = np.arange(20.0).reshape(4, 5)
    artifact = store.put_array("X", array)
    assert store.read_bytes(artifact).endswith(np.ascontiguousarray(array).tobytes())


def test_arrays_that_differ_only_in_shape_are_distinct_artifacts(store: ArtifactStore) -> None:
    values = np.arange(12.0)
    assert store.put_array("x", values.reshape(3, 4)).digest != (
        store.put_array("x", values.reshape(4, 3)).digest
    )


def test_step_records_are_written_under_their_candidate(store: ArtifactStore) -> None:
    record = StepRecord(
        stage="algorithm",
        candidate_id="c1",
        method="gbm",
        backend="python-gbm",
        inputs_digest="abc",
        random_seed=1,
        duration_seconds=0.5,
    )
    path = store.write_step_record("run1", 5, record)
    assert path.relative_to(store.root).parts == (
        "runs",
        "run1",
        "candidates",
        "c1",
        "steps",
        "05-algorithm.json",
    )
    assert StepRecord.model_validate_json(path.read_text()) == record


def test_the_manifest_is_written_as_json(store: ArtifactStore) -> None:
    manifest = RunManifest(
        run_id="run1",
        objective="spatial_transfer",
        data_digest="d",
        package_versions={"numpy": "1.26.0"},
        created_at="2026-08-02T00:00:00+00:00",
    )
    path = store.write_manifest(manifest)
    assert RunManifest.model_validate_json(path.read_text()) == manifest
    assert manifest.is_reproducible


def test_an_artifact_path_is_relative_to_the_store_root(store: ArtifactStore) -> None:
    artifact: Artifact = store.put_json("params", {"n_folds": 4})
    assert not artifact.path.startswith("/")
    assert (store.root / artifact.path).exists()
