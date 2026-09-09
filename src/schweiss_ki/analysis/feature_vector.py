"""AP3 — Featurevektor: eine teilbeschreibende Merkmalszeile pro Bauteil.

Sammelt die vorhandenen, report-getriebenen Kennwerte (`extract_features` +
`fill_volume`) zu einer flachen Zeile pro Bauteil. Kein neuer Rechenweg — nur
einsammeln, gruppieren, benennen. Weil alles aus dem Subtraktions-Report kommt,
funktioniert es für echte Bauteile ohne Codeänderung.

Zweck (siehe Projektziel): der Vektor beschreibt den Ist-Zustand eines Teils und
dient am Ende einem KI-Modell, das daraus Schweißroboter-Parameter an das reale
Teil anpasst. Deshalb teilbeschreibende Merkmale (nicht nur eine Gut/Schlecht-
Klasse) plus Diagnosegrößen zum Vertrauen/Filtern der Zeile.

Drei Spaltengruppen:
  identity   – model_id, Quelle, Label (Soll-Fehler bei synthetischen Fällen)
  features   – physikalische Merkmale (der eigentliche Vektor) inkl. Volumen
  diagnostic – Fit-/Registrier-Qualität, um unsichere Zeilen zu erkennen

Reine Validierungsgrößen (SIGN, expected-vs-measured aus synthetic_validation)
gehören NICHT hierher — der Vektor beschreibt das Teil, er bewertet nicht die
Registrierung.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Optional

import numpy as np

from .synthetic_validation import extract_features
from .weld_volume import DEFAULT_THICKNESS_MM, fill_volume

# ── Spaltengruppen (Reihenfolge = Spaltenreihenfolge im Export) ──────────
IDENTITY_COLUMNS = ["model_id", "source_type", "label", "fault_family"]

FEATURE_COLUMNS = [
    # global
    "global_in_tol_rate", "global_mean_abs",
    # Spalt / Wurzel
    "gap_width_mm", "gap_width_std_mm", "d_root_mm", "gap_wedge_slope",
    # Flankenwinkel (Einzelwinkel sind registrierungsabhängig; die Winkelsumme
    # ist registrierungsinvariant und deshalb das robuste Merkmal)
    "flank_a_angle_deg", "flank_b_angle_deg", "flank_angle_sum_dev_deg",
    "flank_asymmetry_deg",
    # relative Lage der zwei Werkstücke (echtes Fehlausrichtungs-Merkmal)
    "edge_offset_mm", "tilt_total_deg", "tilt_along_seam_deg",
    "tilt_across_gap_deg",
    "measured_tx_mm", "measured_ty_mm", "measured_tz_mm",
    "measured_rx_deg", "measured_ry_deg", "measured_rz_deg",
    # Volumen (AP2.4)
    "fill_volume_mm3", "mean_area_mm2", "area_min_mm2", "area_max_mm2",
    "seam_length_mm",
]

DIAGNOSTIC_COLUMNS = [
    "reg_residual_mm", "reg_converged", "anchored",
    "n_points", "n_bins", "n_bins_valid", "bin_coverage",
    "flank_a_r2", "flank_b_r2", "fit_quality_min_r2",
    "flank_a_rms_mm", "flank_b_rms_mm",
    "ref_plane_inlier_ratio", "ref_plane_rms_mm", "opp_plane_inlier_ratio",
    "thickness_mm", "reinforcement_mm",
]

ALL_COLUMNS = IDENTITY_COLUMNS + FEATURE_COLUMNS + DIAGNOSTIC_COLUMNS


def fault_family(model_id: str, source_type: str) -> str:
    """Grobe Kategorie des Soll-Fehlers aus dem model_id (nur synthetisch)."""
    if source_type != "synthetic":
        return "real"
    m = model_id.upper()
    if m.startswith("T_"):
        return "translation"
    if m.startswith("R_"):
        return "rotation"
    if m.startswith("C_"):
        return "combined"
    return "unknown"


def build_feature_row(report: Dict[str, Any], model_id: str,
                      source_type: str, *,
                      thickness_mm: float = DEFAULT_THICKNESS_MM,
                      reinforcement_mm: float = 0.0,
                      label: Optional[str] = None) -> Dict[str, Any]:
    """Eine Featurezeile für ein Bauteil. Fehlende Größen → NaN."""
    f = extract_features(report)
    vol = fill_volume(report, thickness_mm, reinforcement_mm)

    # Registrierungsinvariante Winkelsumme: (α_A−45) + (α_B−45)
    a, b = f.get("flank_a_angle_deg", math.nan), f.get("flank_b_angle_deg", math.nan)
    wsum = (a - 45.0) + (b - 45.0) if np.isfinite(a) and np.isfinite(b) else math.nan

    row: Dict[str, Any] = {
        "model_id": model_id,
        "source_type": source_type,
        # Label = Soll-Fehler; bei synthetischen Fällen steckt der im model_id.
        "label": (label if label is not None
                  else (model_id if source_type == "synthetic" else "")),
        "fault_family": fault_family(model_id, source_type),
        "flank_angle_sum_dev_deg": wsum,
        "fill_volume_mm3": vol.get("fill_volume_mm3", math.nan),
        "mean_area_mm2": vol.get("mean_area_mm2", math.nan),
        "area_min_mm2": vol.get("area_min_mm2", math.nan),
        "area_max_mm2": vol.get("area_max_mm2", math.nan),
        "seam_length_mm": vol.get("seam_length_mm", math.nan),
        "thickness_mm": float(thickness_mm),
        "reinforcement_mm": float(reinforcement_mm),
    }
    # restliche Merkmale/Diagnose direkt aus extract_features übernehmen
    for col in FEATURE_COLUMNS + DIAGNOSTIC_COLUMNS:
        if col not in row:
            row[col] = f.get(col, math.nan)
    return {col: row.get(col, math.nan) for col in ALL_COLUMNS}
