"""Runtime telemetry helpers for workflow memory and throughput tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass

from sddraft.domain.models import RunMetrics, RunStageMetric

try:  # pragma: no cover - platform-specific import
    import resource
except ImportError:  # pragma: no cover - platform-specific import
    resource = None  # type: ignore[assignment]


def estimate_rss_mb() -> float:
    """Return best-effort resident memory estimate in MB."""

    if resource is None:  # pragma: no cover - platform-specific branch
        return 0.0
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KB, macOS reports bytes.
    if usage > 10_000_000:
        return round(float(usage) / (1024.0 * 1024.0), 3)
    return round(float(usage) / 1024.0, 3)


@dataclass(slots=True)
class _StageWindow:
    stage: str
    started: float
    rss_start: float


class RunMetricsCollector:
    """Collect stage-level metrics with deterministic counters."""

    def __init__(self, csc_id: str) -> None:
        self._metrics = RunMetrics(csc_id=csc_id)
        self._active: _StageWindow | None = None

    def start(self, stage: str) -> None:
        """Start timing a named stage."""

        self._active = _StageWindow(
            stage=stage,
            started=time.perf_counter(),
            rss_start=estimate_rss_mb(),
        )

    def finish(
        self,
        *,
        files_seen: int = 0,
        chunks_written: int = 0,
        chunks_loaded: int = 0,
    ) -> RunStageMetric:
        """Finish active stage and append metric record."""

        if self._active is None:
            raise RuntimeError("No active stage to finish.")

        window = self._active
        self._active = None
        elapsed = round(time.perf_counter() - window.started, 6)
        peak = max(window.rss_start, estimate_rss_mb())
        metric = RunStageMetric(
            stage=window.stage,
            files_seen=files_seen,
            chunks_written=chunks_written,
            chunks_loaded=chunks_loaded,
            elapsed_seconds=elapsed,
            peak_rss_estimate=peak,
        )
        self._metrics.stages.append(metric)
        return metric

    @property
    def metrics(self) -> RunMetrics:
        """Return collected run metrics."""

        return self._metrics
