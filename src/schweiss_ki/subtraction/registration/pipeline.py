"""
RegistrationPipeline – orchestriert eine Sequenz von RegistrationSteps.

Akkumuliert die Delta-Transformationen der einzelnen Steps zur finalen
Source-→-Target-Transformation.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import open3d as o3d
import yaml

from ..base import RegistrationStep
from ..reports import RegistrationReport

logger = logging.getLogger(__name__)


class RegistrationPipeline:
    """Verkettete Ausführung mehrerer RegistrationSteps.

    Nutzung:
        pipeline = RegistrationPipeline([
            CoarsePCA(...),
            ICPFine(...),
        ])
        source_aligned, report = pipeline.run(source, target, source_labels, target_labels)
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
        """Wendet alle Steps nacheinander an.

        Args:
            source:         Real-Scan (wird ausgerichtet).
            target:         CAD-Referenz (fest).
            source_labels:  Optionale Labels für source (aus AP2.1).
            target_labels:  Optionale Labels für target.

        Returns:
            (source_aligned, report):
                - source_aligned: Kopie von source mit finaler Transform.
                - report:         RegistrationReport mit allen Step-Reports
                                  und akkumulierter final_transform.
        """
        if len(source.points) == 0:
            raise ValueError("Source PointCloud ist leer.")
        if len(target.points) == 0:
            raise ValueError("Target PointCloud ist leer.")

        logger.info(
            f"RegistrationPipeline: {len(self.steps)} Steps, "
            f"source={len(source.points):,} pts, target={len(target.points):,} pts"
        )

        # Wir arbeiten auf einer Kopie, damit das Original unangetastet bleibt
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

            # Delta auf source anwenden, kumulierte Transform aktualisieren
            source_aligned.transform(report.delta_transform)
            cumulative = report.delta_transform @ cumulative

            res_str = f", residual={report.residual:.3f}mm" if report.residual is not None else ""
            logger.info(
                f"  ✓ {step.name}: {report.duration_ms:.1f} ms{res_str}"
            )

        total_duration_ms = (time.perf_counter() - t0_total) * 1000.0

        report = RegistrationReport(
            steps=step_reports,
            final_transform=cumulative,
            total_duration_ms=total_duration_ms,
            converged=True,  # TODO: echte Konvergenz-Logik wenn ICP-Step da ist
        )

        logger.info(
            f"RegistrationPipeline fertig: {total_duration_ms:.1f} ms gesamt, "
            f"final residual: "
            f"{report.final_residual:.3f} mm" if report.final_residual is not None else "n/a"
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
                  step_name:
                    enabled: true
                    param1: ...
                    param2: ...

        Reihenfolge der Schlüssel = Ausführungsreihenfolge.

        TODO: Implementierung sobald konkrete Steps existieren.
              Pattern analog zu PreprocessingPipeline.from_config().
        """
        raise NotImplementedError(
            "RegistrationPipeline.from_config() wird implementiert, "
            "sobald konkrete Steps existieren."
        )