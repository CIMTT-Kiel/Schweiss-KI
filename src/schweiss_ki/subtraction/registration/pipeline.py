"""
RegistrationPipeline – orchestriert eine Sequenz von RegistrationSteps.

Akkumuliert die Delta-Transformationen der einzelnen Steps zur finalen
Source-→-Target-Transformation.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import open3d as o3d
import yaml

from ..base import RegistrationStep
from ..reports import RegistrationReport

logger = logging.getLogger(__name__)


# Registry der verfügbaren Step-Klassen für from_config().
# Bei neuem Step: hier eintragen.
def _build_step_registry() -> Dict[str, type]:
    """Lazy import, um Zirkularimporte zu vermeiden."""
    from .coarse_pca import CoarsePCA
    from .x_edge_align import XEdgeAlign
    from .icp_fine import ICPFine
    return {
        "coarse_pca": CoarsePCA,
        "x_edge_align": XEdgeAlign,
        "icp_fine": ICPFine,
    }


class RegistrationPipeline:
    """Verkettete Ausführung mehrerer RegistrationSteps.

    Nutzung:
        pipeline = RegistrationPipeline([
            CoarsePCA(...),
            ICPFine(...),
        ])
        source_aligned, report = pipeline.run(
            source, target, source_labels, target_labels,
        )

    Aus YAML laden:
        pipeline = RegistrationPipeline.from_config(Path("configs/pipeline.yaml"))
    """

    def __init__(self, steps: List[RegistrationStep]):
        self.steps: List[RegistrationStep] = list(steps)

    # ── Ausführung ────────────────────────────────────────────────────

    def run(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        source_labels: Optional[np.ndarray] = None,
        target_labels: Optional[np.ndarray] = None,
    ) -> tuple[o3d.geometry.PointCloud, RegistrationReport]:
        if len(source.points) == 0:
            raise ValueError("Source PointCloud ist leer.")
        if len(target.points) == 0:
            raise ValueError("Target PointCloud ist leer.")

        logger.info(
            f"RegistrationPipeline: {len(self.steps)} Steps, "
            f"source={len(source.points):,} pts, target={len(target.points):,} pts"
        )

        source_aligned = o3d.geometry.PointCloud(source)
        cumulative = np.eye(4)
        step_reports = []

        t0_total = time.perf_counter()

        for step in self.steps:
            if not step.enabled:
                logger.debug(f"  Skip (disabled): {step.name}")
                step_reports.append(
                    step(source_aligned, target, source_labels, target_labels)
                )
                continue

            logger.debug(f"  Run: {step.name}")
            report = step(source_aligned, target, source_labels, target_labels)
            step_reports.append(report)

            source_aligned.transform(report.delta_transform)
            cumulative = report.delta_transform @ cumulative

            res_str = (
                f", residual={report.residual:.3f}mm"
                if report.residual is not None else ""
            )
            logger.info(f"  ✓ {step.name}: {report.duration_ms:.1f} ms{res_str}")

        total_duration_ms = (time.perf_counter() - t0_total) * 1000.0

        report = RegistrationReport(
            steps=step_reports,
            final_transform=cumulative,
            total_duration_ms=total_duration_ms,
            converged=True,
        )

        final_residual = report.final_residual
        if final_residual is not None:
            logger.info(
                f"RegistrationPipeline fertig: {total_duration_ms:.1f} ms gesamt, "
                f"final residual: {final_residual:.3f} mm"
            )
        else:
            logger.info(
                f"RegistrationPipeline fertig: {total_duration_ms:.1f} ms gesamt"
            )

        return source_aligned, report

    # ── Verkettung / Inspektion ───────────────────────────────────────

    def add(self, step: RegistrationStep) -> "RegistrationPipeline":
        self.steps.append(step)
        return self

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        names = ", ".join(s.name for s in self.steps)
        return f"RegistrationPipeline([{names}])"

    # ── YAML-Loading ──────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: Path) -> "RegistrationPipeline":
        """Lädt eine Pipeline aus dem Abschnitt 'subtraction.registration.steps'
        der pipeline.yaml.

        Erwartetes Format:
            subtraction:
              registration:
                steps:
                  coarse_pca:
                    enabled: true
                    anchor_labels: [0, 1, 2]
                  icp_fine:
                    enabled: true
                    max_correspondence_distance: 1.0
                    ...

        Reihenfolge der Schlüssel = Ausführungsreihenfolge (Python 3.7+
        garantiert Insertion-Order in dicts; PyYAML respektiert das).
        """
        config_path = Path(config_path)
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        steps_cfg = (
            cfg.get("subtraction", {})
               .get("registration", {})
               .get("steps", {})
        )
        if not steps_cfg:
            logger.warning(
                "Kein 'subtraction.registration.steps'-Block in Config gefunden – "
                "Pipeline bleibt leer."
            )
            return cls(steps=[])

        registry = _build_step_registry()
        steps: List[RegistrationStep] = []

        for step_name, step_params in steps_cfg.items():
            if step_name not in registry:
                logger.warning(
                    f"Unbekannter Registration-Step '{step_name}' in Config, "
                    f"übersprungen. Verfügbar: {sorted(registry.keys())}"
                )
                continue

            params: Dict[str, Any] = dict(step_params or {})
            step_cls = registry[step_name]
            steps.append(step_cls(**params))

        return cls(steps=steps)