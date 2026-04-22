"""
Basis-Klassen für die Segmentierungs-Pipeline (AP2.1 Phase 3).

Architektur analog zu preprocessing/base.py, mit zwei Unterschieden:
- Die Punktwolke selbst wird nicht modifiziert; Steps schreiben nur labels.
- Jeder Step darf ausschließlich Punkte klassifizieren, die noch UNLABELED (-1)
  sind. Bestehende Klassifikationen vorheriger Steps werden nicht überschrieben.

Datenfluss:
    labels = np.full(n_points, UNLABELED, dtype=np.int8)
    for step in steps:
        labels = step.apply(pcd, labels)
    # Sicherheitsnetz: verbleibende UNLABELED → 0 (background)
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import open3d as o3d

from .labels import LABELS, UNLABELED


# ---------------------------------------------------------------------------
# Report-Datenstrukturen
# ---------------------------------------------------------------------------

@dataclass
class SegmentationStepReport:
    """Bericht eines einzelnen Segmentierungs-Schritts."""

    step_name: str
    enabled: bool
    unlabeled_before: int
    unlabeled_after: int
    assigned: dict[int, int]  # label_id -> Anzahl neu zugewiesener Punkte
    duration_ms: float
    params: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)

    @property
    def points_assigned(self) -> int:
        return sum(self.assigned.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_name": self.step_name,
            "enabled": self.enabled,
            "unlabeled_before": self.unlabeled_before,
            "unlabeled_after": self.unlabeled_after,
            "assigned": {str(k): v for k, v in self.assigned.items()},
            "duration_ms": round(self.duration_ms, 2),
            "params": self.params,
            "artifacts": _artifacts_to_dict(self.artifacts),
        }

    def __repr__(self) -> str:
        assigned_str = ", ".join(
            f"{LABELS.get(lbl, f'?{lbl}')}={n}"
            for lbl, n in self.assigned.items()
        )
        return (
            f"SegmentationStepReport({self.step_name}: "
            f"assigned [{assigned_str}], "
            f"{self.unlabeled_before} → {self.unlabeled_after} unlabeled, "
            f"{self.duration_ms:.1f}ms)"
        )


@dataclass
class SegmentationReport:
    """Vollständiger Bericht einer Pipeline-Ausführung."""

    n_points: int
    points_per_label: dict[int, int]
    total_duration_ms: float
    steps: list[SegmentationStepReport] = field(default_factory=list)

    @property
    def coverage_pct(self) -> float:
        """Anteil klassifizierter Punkte (nicht UNLABELED) in Prozent."""
        if self.n_points == 0:
            return 100.0
        unlabeled = self.points_per_label.get(UNLABELED, 0)
        return 100.0 * (self.n_points - unlabeled) / self.n_points

    def artifacts_of(self, step_name: str) -> dict:
        """Artefakte eines bestimmten Steps (leer wenn Step nicht vorhanden)."""
        for s in self.steps:
            if s.step_name == step_name:
                return s.artifacts
        return {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_points": self.n_points,
            "points_per_label": {str(k): v for k, v in self.points_per_label.items()},
            "total_duration_ms": round(self.total_duration_ms, 2),
            "steps": [s.to_dict() for s in self.steps],
        }

    def summary(self) -> str:
        lines = [
            f"SegmentationReport ({self.n_points:,} Punkte, "
            f"{self.total_duration_ms:.1f}ms, Coverage {self.coverage_pct:.1f}%)",
            f"Klassenverteilung:",
        ]
        for label_id, count in sorted(self.points_per_label.items()):
            if label_id == UNLABELED:
                name = "unlabeled"
            else:
                name = LABELS.get(label_id, f"unknown ({label_id})")
            pct = 100.0 * count / self.n_points if self.n_points > 0 else 0.0
            lines.append(f"  [{label_id}] {name}: {count:,} ({pct:.1f}%)")
        lines.append("Steps:")
        for s in self.steps:
            lines.append(f"  {s}")
        return "\n".join(lines)


def _artifacts_to_dict(artifacts: dict) -> dict:
    """JSON-sichere Konvertierung (numpy arrays → Listen, verschachtelt)."""
    return {k: _coerce_value(v) for k, v in artifacts.items()}


def _coerce_value(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_coerce_value(el) for el in v]
    if isinstance(v, dict):
        return {k: _coerce_value(val) for k, val in v.items()}
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


# ---------------------------------------------------------------------------
# Abstrakte Basisklasse für Segmentierungs-Schritte
# ---------------------------------------------------------------------------

class SegmentationStep(ABC):
    """
    Abstrakte Basisklasse für alle Segmentierungs-Schritte.

    Jeder Step operiert auf Punkten mit labels == UNLABELED (-1).
    Die Punktwolke selbst wird nicht modifiziert.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Eindeutiger Name für Report und Config."""
        ...

    @property
    @abstractmethod
    def enabled(self) -> bool:
        ...

    @abstractmethod
    def _apply(
        self,
        pcd: o3d.geometry.PointCloud,
        labels: np.ndarray,
    ) -> np.ndarray:
        """
        Kernlogik. Gibt ein neues labels-Array zurück.

        Darf nur Punkte mit labels == UNLABELED klassifizieren.
        Bestehende Klassifikationen dürfen nicht verändert werden –
        dies wird in apply() geprüft.
        """
        ...

    @abstractmethod
    def get_params(self) -> dict:
        """Aktuelle Parameter für Report/Logging."""
        ...

    def get_artifacts(self) -> dict:
        """Artefakte der letzten _apply-Ausführung (Ebenen, Winkel, etc.).
        Default: leer. Steps überschreiben bei Bedarf."""
        return {}

    def apply(
        self,
        pcd: o3d.geometry.PointCloud,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, SegmentationStepReport]:
        """Wrappt _apply mit Timing, Diff-Berechnung und Integritätsprüfung."""
        unlabeled_before = int(np.sum(labels == UNLABELED))
        t_start = time.perf_counter()

        if self.enabled:
            labels_out = self._apply(pcd, labels)

            # Integritätsprüfung: bestehende Labels dürfen nicht überschrieben werden
            preexisting = labels != UNLABELED
            if preexisting.any() and not np.array_equal(
                labels[preexisting], labels_out[preexisting]
            ):
                raise RuntimeError(
                    f"Step '{self.name}' hat bestehende Labels überschrieben. "
                    f"Steps dürfen ausschließlich UNLABELED-Punkte klassifizieren."
                )

            artifacts = self.get_artifacts()
        else:
            labels_out = labels
            artifacts = {}

        duration_ms = (time.perf_counter() - t_start) * 1000.0
        unlabeled_after = int(np.sum(labels_out == UNLABELED))

        # Diff: welche Labels wurden in diesem Step neu vergeben?
        changed_mask = labels != labels_out
        assigned: dict[int, int] = {}
        if changed_mask.any():
            new_vals, counts = np.unique(labels_out[changed_mask], return_counts=True)
            assigned = {int(v): int(c) for v, c in zip(new_vals, counts)}

        report = SegmentationStepReport(
            step_name=self.name,
            enabled=self.enabled,
            unlabeled_before=unlabeled_before,
            unlabeled_after=unlabeled_after,
            assigned=assigned,
            duration_ms=duration_ms,
            params=self.get_params() if self.enabled else {},
            artifacts=artifacts,
        )
        return labels_out, report


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class SegmentationPipeline:
    """
    Orchestriert eine geordnete Liste von SegmentationSteps.

    Verwendung (manuell):
        pipeline = SegmentationPipeline([
            BackgroundRemover(...),
            FlankSegmenter(...),
            GapClassifier(...),
        ])
        labels, report = pipeline.process(pcd)
        print(report.summary())

    Verwendung (aus YAML):
        pipeline = SegmentationPipeline.from_config("configs/pipeline.yaml")
        labels, report = pipeline.process(pcd)
    """

    def __init__(
        self,
        steps: Optional[list[SegmentationStep]] = None,
        fill_unlabeled_with_background: bool = True,
    ):
        """
        Args:
            steps: Geordnete Liste von Segmentation-Steps.
            fill_unlabeled_with_background: Wenn True, werden nach allen
                Steps verbleibende UNLABELED-Punkte zu background (0) gesetzt.
        """
        self.steps: list[SegmentationStep] = steps or []
        self.fill_unlabeled_with_background = fill_unlabeled_with_background

    def add_step(self, step: SegmentationStep) -> "SegmentationPipeline":
        self.steps.append(step)
        return self

    def remove_step(self, step_name: str) -> bool:
        before = len(self.steps)
        self.steps = [s for s in self.steps if s.name != step_name]
        return len(self.steps) < before

    def process(
        self,
        pcd: o3d.geometry.PointCloud,
    ) -> tuple[np.ndarray, SegmentationReport]:
        """
        Segmentiert eine (vorverarbeitete) Punktwolke.

        Args:
            pcd: Vorverarbeitete Punktwolke. Voraussetzung: Normalen vorhanden,
                 falls Steps sie benötigen (BackgroundRemover, FlankSegmenter).

        Returns:
            Tuple aus (labels-Array [np.int8, shape=(n_points,)], SegmentationReport)
        """
        n_points = len(pcd.points)
        labels = np.full(n_points, UNLABELED, dtype=np.int8)
        t_start = time.perf_counter()
        step_reports: list[SegmentationStepReport] = []

        for step in self.steps:
            labels, step_report = step.apply(pcd, labels)
            step_reports.append(step_report)

        if self.fill_unlabeled_with_background:
            labels[labels == UNLABELED] = 0  # background

        total_duration_ms = (time.perf_counter() - t_start) * 1000.0

        unique_vals, counts = np.unique(labels, return_counts=True)
        points_per_label = {int(v): int(c) for v, c in zip(unique_vals, counts)}

        report = SegmentationReport(
            n_points=n_points,
            points_per_label=points_per_label,
            total_duration_ms=total_duration_ms,
            steps=step_reports,
        )
        return labels, report

    @classmethod
    def from_config(
        cls,
        config_path: Path | str,
        fill_unlabeled_with_background: bool = True,
    ) -> "SegmentationPipeline":
        """
        Erstellt eine Pipeline aus einer YAML-Konfigurationsdatei.

        Liest den 'segmentation'-Block; respektiert die Reihenfolge der Steps
        wie sie in der Config stehen (YAML dict-Reihenfolge seit Python 3.7).

        Args:
            config_path: Pfad zur YAML-Datei.
            fill_unlabeled_with_background: Siehe __init__.

        Returns:
            Konfigurierte SegmentationPipeline.
        """
        import yaml

        from .background_remover import BackgroundRemover
        from .flank_segmenter import FlankSegmenter
        from .gap_classifier import GapClassifier

        config_path = Path(config_path)
        with open(config_path) as f:
            config = yaml.safe_load(f)

        cfg = config.get("segmentation", config)
        steps_cfg: dict[str, Any] = cfg.get("steps", {})

        step_registry: dict[str, type[SegmentationStep]] = {
            "background_remover": BackgroundRemover,
            "flank_segmenter": FlankSegmenter,
            "gap_classifier": GapClassifier,
        }

        pipeline = cls(fill_unlabeled_with_background=fill_unlabeled_with_background)

        for step_name, step_cfg in steps_cfg.items():
            if not step_cfg.get("enabled", True):
                continue
            if step_name not in step_registry:
                raise ValueError(f"Unbekannter Segmentation-Step: '{step_name}'")
            params = {k: v for k, v in step_cfg.items() if k != "enabled"}
            step = step_registry[step_name](**params)
            pipeline.add_step(step)

        return pipeline

    def __repr__(self) -> str:
        step_names = [s.name for s in self.steps]
        return f"SegmentationPipeline(steps={step_names})"