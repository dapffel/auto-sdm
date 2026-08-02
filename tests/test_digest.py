from __future__ import annotations

import numpy as np
import pytest

from auto_sdm.digest import canonical, digest_bytes, digest_file, digest_obj


def test_identical_content_digests_identically() -> None:
    assert digest_obj({"a": 1}) == digest_obj({"a": 1})


def test_key_order_does_not_change_the_digest() -> None:
    assert digest_obj({"a": 1, "b": 2}) == digest_obj({"b": 2, "a": 1})


def test_array_contents_are_covered() -> None:
    first = np.arange(12.0).reshape(3, 4)
    second = first.copy()
    second[1, 1] += 1e-9
    assert digest_obj(first) != digest_obj(second)


def test_array_shape_is_covered_independently_of_contents() -> None:
    values = np.arange(12.0)
    assert digest_obj(values.reshape(3, 4)) != digest_obj(values.reshape(4, 3))


def test_dtype_is_covered() -> None:
    assert digest_obj(np.arange(4, dtype=np.int32)) != digest_obj(np.arange(4, dtype=np.int64))


def test_a_non_contiguous_view_digests_as_its_values() -> None:
    """A transposed view and its copy hold the same data and must agree."""
    grid = np.arange(12.0).reshape(3, 4)
    assert digest_obj(grid.T) == digest_obj(np.ascontiguousarray(grid.T))


def test_undigestable_types_raise_rather_than_hashing_unstably() -> None:
    class Opaque:
        pass

    with pytest.raises(TypeError, match="cannot digest"):
        digest_obj({"model": Opaque()})


def test_floats_survive_round_tripping() -> None:
    assert digest_obj(0.1 + 0.2) != digest_obj(0.3)
    assert canonical(1.0) != canonical(1)


def test_pydantic_models_are_digestable() -> None:
    from auto_sdm import MethodSpec

    assert digest_obj(MethodSpec(method="gbm")) == digest_obj(MethodSpec(method="gbm"))
    assert digest_obj(MethodSpec(method="gbm")) != digest_obj(MethodSpec(method="logistic"))


def test_file_and_bytes_digests_agree(tmp_path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(b"contents")
    assert digest_file(path) == digest_bytes(b"contents")
