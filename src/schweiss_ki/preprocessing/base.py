"""
Basis-Klassen für die konfigurierbare Preprocessing-Pipeline.

Architektur:
- PreprocessingStep: Abstrakte Basisklasse für alle Preprocessing-Schritte
- StepReport: Bericht eines einzelnen Schritts
- PreprocessingReport: Gesamtbericht einer Pipeline-Ausführung
- PreprocessingPipeline: Orchestriert eine geordnete Liste von Steps

Hinweis (v2 – Zusammenführung):
    report.py und preprocessing/pipeline.py wurden in diese Datei gemergt.
    Beide Dateien können gelöscht werden.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

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

    @property
    def retention_rate(self) -> float:
        """Anteil der behaltenen Punkte (0.0–1.0). Alias für Kompatibilität."""
        if self.points_before == 0:
            return 1.0
        return self.points_after / self.points_before

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_name": self.step_name,
            "enabled": self.enabled,
            "points_before": self.points_before,
            "points_after": self.points_after,
            "duration_ms": round(self.duration_ms, 2),
            "params": self.params,
        }

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
    source_type: Optional[str] = None

    # -- Kern-Properties --------------------------------------------------

    @property
    def points_removed(self) -> int:
        return self.input_points - self.output_points

    @property
    def overall_retention_pct(self) -> float:
        """Behaltene Punkte in Prozent (0–100)."""
        if self.input_points == 0:
            return 100.0
        return 100.0 * self.output_points / self.input_points

    # -- Alias-Properties (Kompatibilität mit pipeline/pipeline.py) -------

    @property
    def points_in(self) -> int:
        return self.input_points

    @property
    def points_out(self) -> int:
        return self.output_points

    @property
    def total_retention_rate(self) -> float:
        """Behaltene Punkte als Rate (0.0–1.0)."""
        if self.input_points == 0:
            return 1.0
        return self.output_points / self.input_points

    # -- Ausgabe -----------------------------------------------------------

    def summary(self) -> str:
        """Lesbare Zusammenfassung für Logs/Notebooks."""
        lines = [
            f"PreprocessingReport",
            f"  Input:  {self.input_points:,} pts",
            f"  Output: {self.output_points:,} pts "
            f"({self.overall_retention_pct:.1f}% retained, "
            f"-{self.points_removed:,} removed)",
            f"  Total:  {self.total_duration_ms:.1f}ms",
        ]
        if self.source_type:
            lines.insert(1, f"  Source: {self.source_type}")
        lines.append("  Steps:")
        for s in self.steps:
            status = "+" if s.enabled else "-"
            lines.append(
                f"    [{status}] {s.step_name:<30} "
                f"{s.points_before:>8,} -> {s.points_after:>8,} pts "
                f"  ({s.duration_ms:.1f}ms)"
            )
        return "\n".join(lines)

    # -- Serialisierung ----------------------------------------------------

    def to_dict(self) -> dict:
        """Serialisierung für JSON-Speicherung im WeldVolumeModel."""
        d: dict[str, Any] = {
            "input_points": self.input_points,
            "output_points": self.output_points,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "steps": [s.to_dict() for s in self.steps],
        }
        if self.source_type is not None:
            d["source_type"] = self.source_type
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "PreprocessingReport":
        """
        Deserialisierung aus JSON (beim Laden eines WeldVolumeModel).

        Akzeptiert sowohl das aktuelle Format (input_points, output_points)
        als auch das alte report.py-Format (source_type + summary-Block).
        """
        # -- Altes report.py-Format erkennen (hat 'summary'-Block) --------
        if "summary" in data and "input_points" not in data:
            summary = data["summary"]
            steps = [
                StepReport(
                    step_name=s["step_name"],
                    enabled=s.get("enabled", True),
                    points_before=s["points_before"],
                    points_after=s["points_after"],
                    duration_ms=s["duration_ms"],
                    params=s.get("params", {}),
                )
                for s in data.get("steps", [])
            ]
            return cls(
                input_points=summary["points_in"],
                output_points=summary["points_out"],
                total_duration_ms=summary["total_duration_ms"],
                steps=steps,
                source_type=data.get("source_type"),
            )

        # -- Aktuelles Format ---------------------------------------------
        steps = [StepReport(**s) for s in data.get("steps", [])]
        return cls(
            input_points=data["input_points"],
            output_points=data["output_points"],
            total_duration_ms=data["total_duration_ms"],
            steps=steps,
            source_type=data.get("source_type"),
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
# Hilfsfunktion
# ---------------------------------------------------------------------------

def _extract_params(step: PreprocessingStep) -> dict[str, Any]:
    """Extrahiert konfigurierbare Parameter aus einem Step für den Report."""
    params = {}
    for attr in vars(step):
        if attr.startswith("_"):
            continue
        val = getattr(step, attr)
        if isinstance(val, (int, float, str, bool, list)):
            params[attr] = val
    return params


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PreprocessingPipeline:
    """
    Orchestriert eine geordnete Liste von PreprocessingSteps.

    Erzeugt nach jeder Ausführung einen vollständigen PreprocessingReport.

    Verwendung (manuell):
        pipeline = PreprocessingPipeline([
            StatisticalOutlierFilter(nb_neighbors=20, std_ratio=2.0),
            VoxelGridDownsampler(voxel_size=0.5),
            NormalEstimator(radius=2.0),
        ])
        pcd_clean, report = pipeline.process(pcd_raw)
        print(report.summary())

    Verwendung (aus YAML):
        pipeline = PreprocessingPipeline.from_config(
            "configs/preprocessing.yaml", source_type="real"
        )
        pcd_clean, report = pipeline.process(pcd_raw)
    """

    def __init__(
        self,
        steps: Optional[list[PreprocessingStep]] = None,
        source_type: Optional[str] = None,
    ):
        self.steps: list[PreprocessingStep] = steps or []
        self.source_type: Optional[str] = source_type

    def add_step(self, step: PreprocessingStep) -> "PreprocessingPipeline":
        """Fügt einen Step hinzu. Gibt self zurück für Method-Chaining."""
        self.steps.append(step)
        return self

    def remove_step(self, step_name: str) -> bool:
        """
        Entfernt einen Step anhand seines Namens.

        Returns:
            True wenn Step gefunden und entfernt, sonst False
        """
        before = len(self.steps)
        self.steps = [s for s in self.steps if s.name != step_name]
        return len(self.steps) < before

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
            source_type=self.source_type,
        )
        return pcd_current, report

    @classmethod
    def from_config(
        cls, config_path: Path | str, source_type: str | None = None
    ) -> "PreprocessingPipeline":
        """
        Erstellt eine Pipeline aus einer YAML-Konfigurationsdatei.

        source_type überschreibt den Wert aus der Config falls angegeben.
        Source-type-Overrides aus der Config werden angewendet.

        Args:
            config_path: Pfad zur YAML-Datei
            source_type: "ideal" | "real" | "synthetic" (überschreibt Config)

        Returns:
            Konfigurierte PreprocessingPipeline
        """
        import yaml

        from .downsampling import RandomDownsampler, VoxelGridDownsampler
        from .filtering import RadiusOutlierFilter, StatisticalOutlierFilter
        from .normalization import Centerer, NormalEstimator

        config_path = Path(config_path)
        with open(config_path) as f:
            config = yaml.safe_load(f)

        cfg = config.get("preprocessing", config)
        resolved_source_type = source_type or cfg.get("source_type", "real")

        # Basis-Steps aus Config laden
        steps_cfg: dict[str, Any] = cfg.get("steps", {})

        # Source-type-spezifische Overrides anwenden
        overrides = cfg.get("source_type_overrides", {}).get(resolved_source_type, {})
        for key, val in overrides.items():
            if key in steps_cfg:
                steps_cfg[key] = {**steps_cfg[key], **val}
            else:
                steps_cfg[key] = val

        # Step-Registry: Name → Klasse
        step_registry: dict[str, type[PreprocessingStep]] = {
            "statistical_outlier_filter": StatisticalOutlierFilter,
            "radius_outlier_filter": RadiusOutlierFilter,
            "voxel_grid_downsampler": VoxelGridDownsampler,
            "random_downsampler": RandomDownsampler,
            "normal_estimator": NormalEstimator,
            "centerer": Centerer,
        }

        pipeline = cls(source_type=resolved_source_type)

        # Reihenfolge aus Config respektieren (YAML dict-Reihenfolge seit Python 3.7)
        for step_name, step_cfg in steps_cfg.items():
            if not step_cfg.get("enabled", True):
                continue
            if step_name not in step_registry:
                raise ValueError(f"Unbekannter Preprocessing-Step: '{step_name}'")

            # Params ohne 'enabled' übergeben
            params = {k: v for k, v in step_cfg.items() if k != "enabled"}
            step = step_registry[step_name](**params)
            pipeline.add_step(step)

        return pipeline

    def __repr__(self) -> str:
        step_names = [s.name for s in self.steps]
        if self.source_type:
            return f"PreprocessingPipeline(source_type='{self.source_type}', steps={step_names})"
        return f"PreprocessingPipeline(steps={step_names})"