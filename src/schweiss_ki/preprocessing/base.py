"""
Basis-Klassen für die konfigurierbare Preprocessing-Pipeline.

Architektur:
- PreprocessingStep: Abstrakte Basisklasse für alle Preprocessing-Schritte
- StepReport: Bericht eines einzelnen Schritts
- PreprocessingReport: Gesamtbericht einer Pipeline-Ausführung
- PreprocessingPipeline: Orchestriert eine geordnete Liste von Steps
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import open3d as o3d


# ---------------------------------------------------------------------------
# Report-Datenstrukturen
# ---------------------------------------------------------------------------

@dataclass
class StepReport:
    """Bericht eines einzelnen Preprocessing-Schritts."""
    step_name: str
    enabled: bool
    points_before: int
    points_after: int
    duration_ms: float
    params: dict = field(default_factory=dict)

    @property
    def points_removed(self) -> int:
        return self.points_before - self.points_after

    @property
    def retention_pct(self) -> float:
        """Anteil der behaltenen Punkte in Prozent."""
        if self.points_before == 0:
            return 100.0
        return 100.0 * self.points_after / self.points_before

    def __repr__(self) -> str:
        return (
            f"StepReport({self.step_name}: "
            f"{self.points_before} → {self.points_after} pts "
            f"[-{self.points_removed}, {self.retention_pct:.1f}%], "
            f"{self.duration_ms:.1f}ms)"
        )


@dataclass
class PreprocessingReport:
    """
    Vollständiger Bericht einer Pipeline-Ausführung.

    Wird direkt im WeldVolumeModel gespeichert, um Reproduzierbarkeit
    und Vorher/Nachher-Vergleiche zu ermöglichen.
    """
    input_points: int
    output_points: int
    total_duration_ms: float
    steps: list[StepReport] = field(default_factory=list)

    @property
    def points_removed(self) -> int:
        return self.input_points - self.output_points

    @property
    def overall_retention_pct(self) -> float:
        if self.input_points == 0:
            return 100.0
        return 100.0 * self.output_points / self.input_points

    def summary(self) -> str:
        """Lesbare Zusammenfassung für Logs/Notebooks."""
        lines = [
            f"PreprocessingReport",
            f"  Input:  {self.input_points:,} pts",
            f"  Output: {self.output_points:,} pts "
            f"({self.overall_retention_pct:.1f}% retained, "
            f"-{self.points_removed:,} removed)",
            f"  Total:  {self.total_duration_ms:.1f}ms",
            f"  Steps:",
        ]
        for s in self.steps:
            status = "+" if s.enabled else "-"
            lines.append(
                f"    [{status}] {s.step_name:<30} "
                f"{s.points_before:>8,} -> {s.points_after:>8,} pts "
                f"  ({s.duration_ms:.1f}ms)"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialisierung für JSON-Speicherung im WeldVolumeModel."""
        return {
            "input_points": self.input_points,
            "output_points": self.output_points,
            "total_duration_ms": self.total_duration_ms,
            "steps": [
                {
                    "step_name": s.step_name,
                    "enabled": s.enabled,
                    "points_before": s.points_before,
                    "points_after": s.points_after,
                    "duration_ms": s.duration_ms,
                    "params": s.params,
                }
                for s in self.steps
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PreprocessingReport":
        """Deserialisierung aus JSON (beim Laden eines WeldVolumeModel)."""
        steps = [StepReport(**s) for s in data.get("steps", [])]
        return cls(
            input_points=data["input_points"],
            output_points=data["output_points"],
            total_duration_ms=data["total_duration_ms"],
            steps=steps,
        )

    def __repr__(self) -> str:
        return (
            f"PreprocessingReport("
            f"{self.input_points:,} -> {self.output_points:,} pts, "
            f"{len(self.steps)} steps, "
            f"{self.total_duration_ms:.1f}ms)"
        )


# ---------------------------------------------------------------------------
# Abstrakte Basisklasse für Preprocessing-Schritte
# ---------------------------------------------------------------------------

class PreprocessingStep(ABC):
    """
    Abstrakte Basisklasse für alle Preprocessing-Schritte.

    Jeder Step ist stateless (Phase 2), aber die save_state/load_state-
    Schnittstelle ist vorbereitet für spätere datengetriebene Schritte.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Eindeutiger Name des Steps für Reports und Config."""
        ...

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Ob der Step aktiv ist. Deaktivierte Steps werden übersprungen."""
        ...

    @abstractmethod
    def _apply(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Kernlogik des Steps. Wird nur aufgerufen wenn enabled=True.

        Args:
            pcd: Eingabe-Punktwolke (nicht modifizieren, neue zurückgeben)

        Returns:
            Verarbeitete Punktwolke
        """
        ...

    @abstractmethod
    def get_params(self) -> dict:
        """Aktuelle Parameter für Report/Logging."""
        ...

    def apply(self, pcd: o3d.geometry.PointCloud) -> tuple[o3d.geometry.PointCloud, StepReport]:
        """
        Wendet den Step an und gibt Punktwolke + Report zurück.

        Übernimmt Timing und Punkt-Zählung automatisch.
        Wenn enabled=False, wird die Punktwolke unverändert durchgereicht.
        """
        points_before = len(pcd.points)
        t_start = time.perf_counter()

        if self.enabled:
            pcd_out = self._apply(pcd)
        else:
            pcd_out = pcd

        duration_ms = (time.perf_counter() - t_start) * 1000.0
        points_after = len(pcd_out.points)

        report = StepReport(
            step_name=self.name,
            enabled=self.enabled,
            points_before=points_before,
            points_after=points_after,
            duration_ms=duration_ms,
            params=self.get_params() if self.enabled else {},
        )
        return pcd_out, report

    # ------------------------------------------------------------------
    # State-Schnittstelle (aktuell No-Ops; für spätere ML-Schritte)
    # ------------------------------------------------------------------

    def save_state(self, path: Path) -> None:
        """Speichert gelernten State (z.B. Filterparameter aus Trainingsdaten).
        Aktuell No-Op für alle stateless Steps."""
        pass

    def load_state(self, path: Path) -> None:
        """Lädt gespeicherten State. Aktuell No-Op für alle stateless Steps."""
        pass


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PreprocessingPipeline:
    """
    Orchestriert eine geordnete Liste von PreprocessingSteps.

    Erzeugt nach jeder Ausführung einen vollständigen PreprocessingReport.

    Beispiel:
        pipeline = PreprocessingPipeline([
            StatisticalOutlierFilter(nb_neighbors=20, std_ratio=2.0),
            VoxelGridDownsampler(voxel_size=0.5),
            NormalEstimator(radius=2.0),
        ])
        pcd_clean, report = pipeline.process(pcd_raw)
        print(report.summary())
    """

    def __init__(self, steps: Optional[list[PreprocessingStep]] = None):
        self.steps: list[PreprocessingStep] = steps or []

    def add_step(self, step: PreprocessingStep) -> "PreprocessingPipeline":
        """Fügt einen Step hinzu. Gibt self zurück für Method-Chaining."""
        self.steps.append(step)
        return self

    def process(
        self, pcd: o3d.geometry.PointCloud
    ) -> tuple[o3d.geometry.PointCloud, PreprocessingReport]:
        """
        Führt alle Steps sequenziell aus.

        Args:
            pcd: Roh-Punktwolke

        Returns:
            Tuple aus (bereinigter Punktwolke, PreprocessingReport)
        """
        input_points = len(pcd.points)
        t_start = time.perf_counter()

        step_reports: list[StepReport] = []
        pcd_current = pcd

        for step in self.steps:
            pcd_current, step_report = step.apply(pcd_current)
            step_reports.append(step_report)

        total_duration_ms = (time.perf_counter() - t_start) * 1000.0

        report = PreprocessingReport(
            input_points=input_points,
            output_points=len(pcd_current.points),
            total_duration_ms=total_duration_ms,
            steps=step_reports,
        )
        return pcd_current, report

    def __repr__(self) -> str:
        step_names = [s.name for s in self.steps]
        return f"PreprocessingPipeline(steps={step_names})"