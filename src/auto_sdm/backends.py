"""The backend registry.

Backends are *capability providers*, not a language choice. A methodology names a method;
the registry resolves it to whichever backend implements it — Python, R, or synthesized.
A method nothing implements resolves to an ``UnsupportedStep``, never to a substitute.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .models import Assumption, MethodSpec, UnsupportedStep

Runtime = str  # "python" | "r" | "synthesized"


@runtime_checkable
class Backend(Protocol):
    """One executable method.

    Implementations must be deterministic given ``random_seed``: the run record claims
    reproducibility, and a backend that ignores the seed silently breaks that claim.
    """

    name: str
    stage: str
    methods: tuple[str, ...]
    runtime: Runtime

    def run(
        self,
        spec: MethodSpec,
        inputs: dict[str, Any],
        random_seed: int,
    ) -> tuple[dict[str, Any], list[Assumption]]:
        """Execute the method, returning its outputs and any fallbacks it made."""
        ...


class Resolution:
    """The outcome of looking a method up: either a backend, or a recorded gap."""

    __slots__ = ("backend", "gap")

    def __init__(self, backend: Backend | None, gap: UnsupportedStep | None) -> None:
        if (backend is None) == (gap is None):
            raise ValueError("a resolution is exactly one of a backend or a gap")
        self.backend = backend
        self.gap = gap

    @property
    def ok(self) -> bool:
        return self.backend is not None


class BackendRegistry:
    """Maps (stage, method) to a backend.

    Registration is keyed on the *method* name rather than the backend, so an R backend
    and a synthesized backend are indistinguishable to the executor.
    """

    def __init__(self) -> None:
        self._backends: dict[tuple[str, str], Backend] = {}

    def register(self, backend: Backend) -> None:
        for method in backend.methods:
            key = (backend.stage, method.lower())
            existing = self._backends.get(key)
            if existing is not None and existing.name != backend.name:
                raise ValueError(f"{backend.stage}/{method} already provided by {existing.name}")
            self._backends[key] = backend

    def resolve(self, stage: str, spec: MethodSpec) -> Resolution:
        backend = self._backends.get((stage, spec.method.lower()))
        if backend is not None:
            return Resolution(backend, None)
        return Resolution(
            None,
            UnsupportedStep(
                code=f"unsupported_{stage}",
                message=f"no registered backend implements {stage!r} method {spec.method!r}",
                feature=stage,
                requested_value=spec.method,
            ),
        )

    def methods_for(self, stage: str) -> list[str]:
        """Every method name registered for a stage, including aliases."""
        return sorted(
            method for registered_stage, method in self._backends if registered_stage == stage
        )

    def backends_for(self, stage: str) -> list[Backend]:
        """The distinct capabilities registered for a stage.

        The proposer enumerates over these rather than over ``methods_for``: 'gbm' and
        'brt' name one backend, and treating them as two points in the search space would
        spend half the compute running the same candidate twice.
        """
        seen: dict[str, Backend] = {}
        for (registered_stage, _), backend in self._backends.items():
            if registered_stage == stage:
                seen.setdefault(backend.name, backend)
        return [seen[name] for name in sorted(seen)]

    def canonical_method(self, backend: Backend) -> str:
        """The name a proposed plan should use for a backend: its first declared method."""
        return backend.methods[0]

    def runtimes(self) -> dict[str, Runtime]:
        return {backend.name: backend.runtime for backend in set(self._backends.values())}
