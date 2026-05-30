"""Shared eval-framework primitives: baselines, drift comparison, registry.

Each behavioral eval module (band_routing.py, tempo_accuracy.py, etc.)
exposes:

    NAME: str              # category identifier (matches baseline filename)
    TOLERANCES: dict       # per-metric relative tolerance for regression check
    def run() -> dict[str, float]:
        ...                # produce headline metrics

The regression test loads each category's baseline JSON, calls run(),
and asserts no metric has drifted further than its tolerance. The
refresh tool calls run() and rewrites the baseline.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASELINE_DIR = Path(__file__).parent / 'baselines'


@dataclass
class BaselineMetadata:
    """Provenance for a baseline snapshot.

    git_commit: short SHA at time of baseline capture. Lets diff readers
        tie a baseline change to the commit it was produced against.
    timestamp_utc: ISO-format UTC time. Lets stale baselines be spotted.
    refresh_note: optional human-readable reason for the refresh —
        e.g. "after switching to PercentileBeatDetector default".
    """
    git_commit: str | None = None
    timestamp_utc: str = ''
    refresh_note: str = ''


@dataclass
class Baseline:
    """One eval category's checked-in expected behavior.

    metrics maps headline metric names (e.g. 'recall_low') to baseline
    values. Drift is measured per metric; tolerances live in the
    category module so the baseline file itself stays "what the
    algorithm currently does," not "what we're willing to accept."
    """
    name: str
    metrics: dict[str, float] = field(default_factory=dict)
    per_stimulus: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: BaselineMetadata = field(default_factory=BaselineMetadata)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Baseline:
        meta = d.get('metadata', {})
        return cls(
            name=d['name'],
            metrics=dict(d.get('metrics', {})),
            per_stimulus=dict(d.get('per_stimulus', {})),
            metadata=BaselineMetadata(
                git_commit=meta.get('git_commit'),
                timestamp_utc=meta.get('timestamp_utc', ''),
                refresh_note=meta.get('refresh_note', ''),
            ),
        )


def baseline_path(name: str) -> Path:
    """Where category `name`'s baseline JSON lives."""
    return BASELINE_DIR / f'{name}.json'


def load_baseline(name: str) -> Baseline | None:
    """Load a checked-in baseline. Returns None if not present — useful
    for the first run of a new category."""
    p = baseline_path(name)
    if not p.exists():
        return None
    return Baseline.from_dict(json.loads(p.read_text()))


def save_baseline(baseline: Baseline) -> None:
    """Write the baseline JSON, creating the directory if needed.

    Stable key ordering and trailing newline so diffs read cleanly and
    the file's content hash is reproducible across machines.
    """
    p = baseline_path(baseline.name)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(baseline.to_dict(), indent=2, sort_keys=True)
    p.write_text(text + '\n')


def _current_git_commit() -> str | None:
    """Short git SHA at HEAD. Returns None if not in a repo / git missing.
    Called by the refresh tool when capturing a new baseline."""
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL, text=True, timeout=2)
        return out.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def make_metadata(refresh_note: str = '') -> BaselineMetadata:
    return BaselineMetadata(
        git_commit=_current_git_commit(),
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        refresh_note=refresh_note,
    )


@dataclass
class MetricDrift:
    """One metric's comparison against baseline."""
    name: str
    current: float
    baseline: float
    tolerance_rel: float       # accepted relative drift (e.g. 0.10 = 10%)
    delta_abs: float
    delta_rel: float           # delta / max(|baseline|, 1e-9)
    within_tolerance: bool


@dataclass
class DriftReport:
    """Per-metric drift summary. drifted is non-empty iff any metric
    exceeded its tolerance — that's the test-failure signal."""
    name: str
    within_tolerance: list[MetricDrift] = field(default_factory=list)
    drifted: list[MetricDrift] = field(default_factory=list)
    missing_from_current: list[str] = field(default_factory=list)
    missing_from_baseline: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (not self.drifted
                and not self.missing_from_current
                and not self.missing_from_baseline)

    def summary(self) -> str:
        """Human-readable summary, suitable for test failure messages."""
        lines = [f'Eval drift for {self.name!r}:']
        if self.drifted:
            lines.append(f'  {len(self.drifted)} drifted metric(s):')
            for d in self.drifted:
                pct = 100.0 * d.delta_rel
                lines.append(
                    f'    {d.name}: current={d.current:.4f}  '
                    f'baseline={d.baseline:.4f}  '
                    f'drift={pct:+.1f}%  (tolerance ±{100*d.tolerance_rel:.1f}%)')
        if self.missing_from_current:
            lines.append(
                f'  metrics in baseline but not produced this run: '
                f'{self.missing_from_current}')
        if self.missing_from_baseline:
            lines.append(
                f'  new metrics produced (not in baseline): '
                f'{self.missing_from_baseline}')
        if self.ok:
            lines.append(f'  all {len(self.within_tolerance)} metric(s) '
                          f'within tolerance')
        return '\n'.join(lines)


def compare(name: str, current: dict[str, float], baseline: Baseline,
            tolerances: dict[str, float]) -> DriftReport:
    """Per-metric drift comparison.

    tolerances: per-metric *relative* tolerance (0.10 = ±10% of
    baseline value allowed). Metrics not in `tolerances` use a default
    of 0.05 (5%) — pick small for tight categories, large for known-
    noisy ones.
    """
    report = DriftReport(name=name)
    default_tol = tolerances.get('_default', 0.05)

    cur_keys = set(current.keys())
    base_keys = set(baseline.metrics.keys())

    for k in sorted(cur_keys | base_keys):
        if k not in cur_keys:
            report.missing_from_current.append(k)
            continue
        if k not in base_keys:
            report.missing_from_baseline.append(k)
            continue
        cur = float(current[k])
        base = float(baseline.metrics[k])
        delta_abs = cur - base
        delta_rel = delta_abs / max(abs(base), 1e-9)
        tol = tolerances.get(k, default_tol)
        drift = MetricDrift(
            name=k, current=cur, baseline=base,
            tolerance_rel=tol,
            delta_abs=delta_abs, delta_rel=delta_rel,
            within_tolerance=abs(delta_rel) <= tol,
        )
        if drift.within_tolerance:
            report.within_tolerance.append(drift)
        else:
            report.drifted.append(drift)
    return report


# ---------------------------------------------------------------------------
# Category registry
# ---------------------------------------------------------------------------
#
# Each category module gets registered here so the refresh tool and the
# regression test can iterate over all known categories without each
# being aware of every module name. New categories: add an import + a
# REGISTRY entry.

class _LazyCategory:
    """Defers the actual import until something asks for run() — keeps
    `import flame_sheep_audio.eval` cheap, since each category may pull
    in numpy / AudioProcessor / sample synthesis."""
    def __init__(self, module_name: str):
        self._module_name = module_name
        self._module = None

    def _load(self):
        if self._module is None:
            import importlib
            self._module = importlib.import_module(
                f'flame_sheep_audio.eval.{self._module_name}')
        return self._module

    @property
    def name(self) -> str:
        return self._load().NAME

    @property
    def tolerances(self) -> dict[str, float]:
        return self._load().TOLERANCES

    def run(self) -> tuple[dict[str, float], dict[str, Any]]:
        """Returns (headline_metrics, per_stimulus_details)."""
        return self._load().run()


# Add new categories here when they ship.
REGISTRY: list[_LazyCategory] = [
    _LazyCategory('band_routing'),
]
