from __future__ import annotations

from typing import Any

import pytest

from auto_sdm import Assumption, BackendRegistry, MethodSpec
from auto_sdm.backends import Runtime


class StubBackend:
    def __init__(
        self, name: str, stage: str, methods: tuple[str, ...], runtime: Runtime = "python"
    ) -> None:
        self.name = name
        self.stage = stage
        self.methods = methods
        self.runtime = runtime

    def run(
        self, spec: MethodSpec, inputs: dict[str, Any], random_seed: int
    ) -> tuple[dict[str, Any], list[Assumption]]:
        return {"fitted": spec.method, "seed": random_seed}, []


def test_a_registered_method_resolves_to_its_backend() -> None:
    registry = BackendRegistry()
    registry.register(StubBackend("python-maxent", "algorithm", ("maxent",)))
    resolution = registry.resolve("algorithm", MethodSpec(method="maxent"))
    assert resolution.ok
    assert resolution.backend is not None
    assert resolution.backend.name == "python-maxent"


def test_an_unregistered_method_becomes_a_capability_gap_not_an_error() -> None:
    """The load-bearing behaviour: never substitute, always record."""
    registry = BackendRegistry()
    resolution = registry.resolve("algorithm", MethodSpec(method="brt"))
    assert not resolution.ok
    assert resolution.gap is not None
    assert resolution.gap.feature == "algorithm"
    assert resolution.gap.requested_value == "brt"


def test_gaps_carry_a_stable_code_so_they_aggregate_across_runs() -> None:
    registry = BackendRegistry()
    first = registry.resolve("split", MethodSpec(method="buffered_loo"))
    second = registry.resolve("split", MethodSpec(method="environmental_block"))
    assert first.gap is not None and second.gap is not None
    assert first.gap.code == second.gap.code == "unsupported_split"


def test_resolution_is_case_insensitive() -> None:
    registry = BackendRegistry()
    registry.register(StubBackend("python-maxent", "algorithm", ("maxent",)))
    assert registry.resolve("algorithm", MethodSpec(method="MaxEnt")).ok


def test_the_same_method_in_different_stages_does_not_collide() -> None:
    registry = BackendRegistry()
    registry.register(StubBackend("split-random", "split", ("random",)))
    registry.register(StubBackend("bg-random", "background", ("random",)))
    split = registry.resolve("split", MethodSpec(method="random"))
    background = registry.resolve("background", MethodSpec(method="random"))
    assert split.backend is not None and background.backend is not None
    assert split.backend.name != background.backend.name


def test_two_backends_may_not_claim_the_same_method() -> None:
    registry = BackendRegistry()
    registry.register(StubBackend("python-gam", "algorithm", ("gam",)))
    with pytest.raises(ValueError, match="already provided by"):
        registry.register(StubBackend("r-mgcv", "algorithm", ("gam",), runtime="r"))


def test_r_and_python_backends_are_indistinguishable_to_the_executor() -> None:
    registry = BackendRegistry()
    registry.register(StubBackend("python-maxent", "algorithm", ("maxent",)))
    registry.register(StubBackend("r-earth", "algorithm", ("mars",), runtime="r"))
    for method in ("maxent", "mars"):
        resolution = registry.resolve("algorithm", MethodSpec(method=method))
        assert resolution.ok


def test_the_registry_reports_the_executable_search_space() -> None:
    """What the proposer is allowed to choose from."""
    registry = BackendRegistry()
    registry.register(StubBackend("python-maxent", "algorithm", ("maxent",)))
    registry.register(StubBackend("python-gbm", "algorithm", ("brt", "gbm")))
    registry.register(StubBackend("split-random", "split", ("random",)))
    assert registry.methods_for("algorithm") == ["brt", "gbm", "maxent"]
    assert registry.methods_for("split") == ["random"]


def test_a_resolution_is_exactly_one_of_a_backend_or_a_gap() -> None:
    from auto_sdm.backends import Resolution

    with pytest.raises(ValueError):
        Resolution(None, None)
