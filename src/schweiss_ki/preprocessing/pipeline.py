"""
Konfigurierbare Preprocessing-Pipeline.

Orchestriert eine geordnete Liste von PreprocessingSteps und erstellt
automatisch einen PreprocessingReport für das WeldVolumeModel.
"""
import time
from pathlib import Path
from typing import Any

import open3d as o3d

from .base import PreprocessingStep
from .report import PreprocessingReport, StepReport


class PreprocessingPipeline:
    """
    Konfigurierbare Preprocessing-Pipeline für Punktwolken.

    Verwaltet eine geordnete Liste von PreprocessingSteps und führt sie
    sequenziell aus. Für jeden Step wird ein StepReport erstellt, der
    zu einem PreprocessingReport zusammengefasst wird.

    Typische Verwendung:
        pipeline = PreprocessingPipeline.from_config("configs/pipeline.yaml", source_type="real")
        pcd_clean, report = pipeline.process(pcd_raw)

    Oder manuell:
        pipeline = PreprocessingPipeline(source_type="real")
        pipeline.add_step(StatisticalOutlierFilter(nb_neighbors=20, std_ratio=2.0))
        pipeline.add_step(VoxelGridDownsampler(voxel_size=0.5))
        pcd_clean, report = pipeline.process(pcd_raw)
    """

    def __init__(self, source_type: str = "real"):
        self.source_type = source_type
        self._steps: list[PreprocessingStep] = []

    def add_step(self, step: PreprocessingStep) -> "PreprocessingPipeline":
        """Fügt einen Step ans Ende der Pipeline an. Gibt self zurück für Method Chaining."""
        self._steps.append(step)
        return self

    def remove_step(self, step_name: str) -> bool:
        """
        Entfernt einen Step anhand seines Namens.

        Returns:
            True wenn Step gefunden und entfernt, sonst False
        """
        before = len(self._steps)
        self._steps = [s for s in self._steps if s.name != step_name]
        return len(self._steps) < before

    @property
    def steps(self) -> list[PreprocessingStep]:
        return list(self._steps)

    def process(
        self, pcd: o3d.geometry.PointCloud
    ) -> tuple[o3d.geometry.PointCloud, PreprocessingReport]:
        """
        Führt alle Steps sequenziell aus.

        Args:
            pcd: Eingabe-Punktwolke

        Returns:
            Tuple aus (verarbeitete Punktwolke, PreprocessingReport)
        """
        report = PreprocessingReport(source_type=self.source_type)
        current_pcd = pcd

        for step in self._steps:
            points_before = len(current_pcd.points)
            t_start = time.perf_counter()

            # apply() gibt (PointCloud, base.StepReport) zurück – Tuple entpacken!
            current_pcd, base_report = step.apply(current_pcd)

            duration_ms = (time.perf_counter() - t_start) * 1000

            step_report = StepReport(
                step_name=step.name,
                params=base_report.params,
                points_before=points_before,
                points_after=len(current_pcd.points),
                duration_ms=duration_ms,
            )
            report.steps.append(step_report)

        return current_pcd, report

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
        from .normalization import NormalEstimator

        config_path = Path(config_path)
        with open(config_path) as f:
            config = yaml.safe_load(f)

        cfg = config.get("preprocessing", config)
        resolved_source_type = source_type or cfg.get("source_type", "real")

        # Basis-Steps aus Config laden
        steps_cfg: dict[str, Any] = dict(cfg.get("steps", {}))

        # Source-type-spezifische Overrides anwenden
        overrides = cfg.get("source_type_overrides", {}).get(resolved_source_type, {})
        for key, val in overrides.items():
            if key in steps_cfg:
                steps_cfg[key] = {**steps_cfg[key], **val}
            else:
                steps_cfg[key] = val

        step_registry = {
            "statistical_outlier_filter": StatisticalOutlierFilter,
            "radius_outlier_filter": RadiusOutlierFilter,
            "voxel_grid_downsampler": VoxelGridDownsampler,
            "random_downsampler": RandomDownsampler,
            "normal_estimator": NormalEstimator,
        }

        pipeline = cls(source_type=resolved_source_type)

        # Reihenfolge aus Config respektieren (YAML dict-Reihenfolge seit Python 3.7)
        for step_name, step_cfg in steps_cfg.items():
            if not step_cfg.get("enabled", True):
                continue
            if step_name not in step_registry:
                raise ValueError(f"Unbekannter Preprocessing-Step: '{step_name}'")

            params = {k: v for k, v in step_cfg.items() if k != "enabled"}
            pipeline.add_step(step_registry[step_name](**params))

        return pipeline

    def __repr__(self) -> str:
        step_names = [s.name for s in self._steps]
        return f"PreprocessingPipeline(source_type='{self.source_type}', steps={step_names})"