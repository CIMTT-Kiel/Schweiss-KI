"""
Report-Datenklassen für das Subtraction-Modul (AP2.2).

Hierarchie:
    SubtractionReport
    ├── registration:  RegistrationReport
    │   └── steps:     List[RegistrationStepReport]
    └── deviation:     DeviationData
        └── step_reports: List[DeviationStepReport]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# Registration Reports
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class RegistrationStepReport:
    """Report eines einzelnen Registrierungs-Schritts.

    Attributes:
        step_name:        Name des Steps (z.B. "coarse_pca", "icp_fine").
        enabled:          Ob der Step ausgeführt wurde.
        duration_ms:      Laufzeit des Steps in ms.
        delta_transform:  4×4-Transformationsmatrix, die DIESER Step beigetragen hat.
        residual:         Residuum nach Anwendung des Steps (z.B. RMSE in mm).
        fitness:          Optionales Gütemaß (z.B. ICP-Fitness, Anteil Korrespondenzen).
        params:           Parameter, mit denen der Step lief.
        artifacts:        Schritt-spezifische Zwischenergebnisse (für Debugging/Plots).
    """
    step_name: str
    enabled: bool = True
    duration_ms: float = 0.0
    delta_transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    residual: Optional[float] = None
    fitness: Optional[float] = None
    params: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "step_name": self.step_name,
            "enabled": self.enabled,
            "duration_ms": self.duration_ms,
            "delta_transform": self.delta_transform.tolist(),
            "residual": self.residual,
            "fitness": self.fitness,
            "params": self.params,
            # artifacts werden bewusst nicht serialisiert (können numpy-Arrays etc. enthalten)
        }


@dataclass
class RegistrationReport:
    """Gesamtreport einer Registrierungs-Pipeline.

    Attributes:
        steps:              Liste aller Step-Reports in Ausführungsreihenfolge.
        final_transform:    Akkumulierte 4×4-Transformation (Produkt aller Delta-Transforms).
        total_duration_ms:  Gesamtlaufzeit der Pipeline.
        converged:          Hat die Pipeline ein Konvergenzkriterium erreicht
                            (nur sinnvoll, wenn alle Steps liefen).
    """
    steps: List[RegistrationStepReport] = field(default_factory=list)
    final_transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    total_duration_ms: float = 0.0
    converged: bool = False

    @property
    def final_residual(self) -> Optional[float]:
        """Residuum nach dem letzten Step mit gesetztem Residuum."""
        for s in reversed(self.steps):
            if s.residual is not None:
                return s.residual
        return None

    def summary(self) -> str:
        lines = [
            f"RegistrationReport – {len(self.steps)} Steps, "
            f"{self.total_duration_ms:.1f} ms gesamt",
        ]
        for s in self.steps:
            status = "✓" if s.enabled else "·"
            res = f"  residual={s.residual:.3f}mm" if s.residual is not None else ""
            fit = f"  fitness={s.fitness:.3f}" if s.fitness is not None else ""
            lines.append(
                f"  {status} {s.step_name:24s}  {s.duration_ms:7.1f} ms{res}{fit}"
            )
        final_res = self.final_residual
        if final_res is not None:
            lines.append(f"  → final residual: {final_res:.3f} mm")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "final_transform": self.final_transform.tolist(),
            "total_duration_ms": self.total_duration_ms,
            "converged": self.converged,
            "final_residual": self.final_residual,
        }


# ─────────────────────────────────────────────────────────────────────────
# Deviation Reports
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class DeviationStepReport:
    """Report eines einzelnen Deviation-Schritts."""
    step_name: str
    enabled: bool = True
    duration_ms: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "step_name": self.step_name,
            "enabled": self.enabled,
            "duration_ms": self.duration_ms,
            "params": self.params,
        }


@dataclass
class DeviationData:
    """Ergebnis der Differenzbildanalyse.

    Attributes:
        distances_signed:    Vorzeichenbehaftete Distanz pro Source-Punkt (mm).
                             Positiv = über CAD-Oberfläche, negativ = darunter.
                             Konvention abhängig von der CAD-Normale.
        distances_absolute:  |distances_signed|, zur Bequemlichkeit.
        in_tolerance:        Bool-Flag pro Punkt: |signed| <= tolerance_mm.
        tolerance_mm:        Verwendete Toleranzschwelle.
        per_region_metrics:  Aggregierte Metriken pro Segmentierungs-Label.
                             Schlüssel = Label-ID, Wert = dict mit "mean",
                             "std", "max_abs", "in_tolerance_rate", etc.
        gap_profile:         Optionales Profil entlang der Naht (Spaltmaß(Y),
                             Wurzeltiefe(Y), Öffnungswinkel(Y) etc.).
        step_reports:        Reports der einzelnen Deviation-Steps.
        total_duration_ms:   Gesamtlaufzeit.
    """
    distances_signed: np.ndarray = field(default_factory=lambda: np.zeros(0))
    in_tolerance: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    tolerance_mm: float = 0.25
    per_region_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)
    gap_profile: Optional[Dict[str, np.ndarray]] = None
    step_reports: List[DeviationStepReport] = field(default_factory=list)
    total_duration_ms: float = 0.0

    @property
    def distances_absolute(self) -> np.ndarray:
        return np.abs(self.distances_signed)

    @property
    def overall_in_tolerance_rate(self) -> float:
        if len(self.in_tolerance) == 0:
            return 0.0
        return float(self.in_tolerance.mean())

    def summary(self) -> str:
        n = len(self.distances_signed)
        if n == 0:
            return "DeviationData – leer"
        d_abs = self.distances_absolute
        lines = [
            f"DeviationData – {n:,} Punkte, Toleranz ±{self.tolerance_mm} mm",
            f"  |d|: mean={d_abs.mean():.3f} mm, "
            f"max={d_abs.max():.3f} mm, p95={np.percentile(d_abs, 95):.3f} mm",
            f"  in tolerance: {self.overall_in_tolerance_rate*100:.1f} %",
        ]
        for label_id, metrics in sorted(self.per_region_metrics.items()):
            mean_d = metrics.get("mean_abs", float("nan"))
            in_tol = metrics.get("in_tolerance_rate", float("nan"))
            lines.append(
                f"  Label {label_id}: |d|={mean_d:.3f} mm, "
                f"in tol={in_tol*100:.1f} %"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "n_points": len(self.distances_signed),
            "tolerance_mm": self.tolerance_mm,
            "overall_in_tolerance_rate": self.overall_in_tolerance_rate,
            "per_region_metrics": self.per_region_metrics,
            "step_reports": [s.to_dict() for s in self.step_reports],
            "total_duration_ms": self.total_duration_ms,
            # distances_signed/in_tolerance werden separat als .npy gespeichert
        }


# ─────────────────────────────────────────────────────────────────────────
# Top-level Report
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class SubtractionReport:
    """Gesamtreport der Subtraction-Methode (AP2.2).

    Bündelt Registrierung und Differenzbildanalyse für das WeldVolumeModel.
    """
    registration: RegistrationReport = field(default_factory=RegistrationReport)
    deviation: DeviationData = field(default_factory=DeviationData)
    cad_source_file: Optional[str] = None

    def summary(self) -> str:
        return "\n\n".join([
            self.registration.summary(),
            self.deviation.summary(),
        ])

    def to_dict(self) -> dict:
        return {
            "cad_source_file": self.cad_source_file,
            "registration": self.registration.to_dict(),
            "deviation": self.deviation.to_dict(),
        }