"""Merkmalsextraktion aus den Batch-Reports der synthetischen Serie.

Liest `subtraction_report.json` je Fall und fasst die drei Auswertungsebenen
zu einer Zeile je Testfall zusammen:

* **global**    – Distanzkennwerte über alle Punkte und je Region
* **lokal**     – Voxel-Aggregate (hier verdichtet; die Rohgitter bleiben im Report)
* **Merkmale**  – Flankenwinkel, Asymmetrie, Kantenversatz, Verkippung, Fit-Güte

Die Ground Truth der synthetischen Fälle steht in `synthetic_metadata.csv` und
beschreibt, wie Werkstück B gegenüber A verstellt wurde. Gemessen wird die
Relativlage in der Gegenrichtung, siehe `SIGN`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# Nennmaße des Referenzbauteils (Baugruppe_Beispielteil_V-Naht_1.5mm_Spalt).
NOMINAL_ROOT_GAP_MM = 1.5
NOMINAL_THICKNESS_MM = 5.0
NOMINAL_FLANK_ANGLE_DEG = 45.0
TOLERANCE_MM = 0.25

# Die Komponenten-Vermessung liefert die Lage von A relativ zu B, der
# Generator verstellt B gegenüber A. Für Translationen dreht sich das
# Vorzeichen daher in X und Z; Y ist durch die Spaltöffnung anders gekoppelt.
SIGN = {"tx_mm": -1, "ty_mm": +1, "tz_mm": -1,
        "rx_deg": -1, "ry_deg": -1, "rz_deg": -1}

# Die neun Generator-Kategorien auf die drei Fehlerklassen abbilden, nach
# denen ausgewertet wird.
ERROR_CLASS = {
    "translation_x": "Translation",
    "translation_y": "Translation",
    "translation_z": "Translation",
    "translation_combo": "Translation",
    "rotation_x": "Rotation",
    "rotation_y": "Rotation",
    "rotation_z": "Rotation",
    "rotation_combo": "Rotation",
    "translation_rotation_combo": "Kombination",
}
CLASS_ORDER = ["Translation", "Rotation", "Kombination"]
CLASS_COLORS = {"Translation": "#1976D2",
                "Rotation": "#FF9800",
                "Kombination": "#43A047"}


def _finite(values) -> np.ndarray:
    """Nur endliche Einträge. JSON transportiert fehlende Bins als NaN/None."""
    arr = np.array([np.nan if v is None else v for v in values], dtype=float)
    return arr[np.isfinite(arr)]


def _nanmean(values) -> float:
    arr = _finite(values)
    return float(arr.mean()) if arr.size else math.nan


def slope_to_angle_deg(slope: float) -> float:
    """Flankensteigung dy/dd in einen Winkel zur Vertikalen umrechnen.

    Das Profil beschreibt die Flanke als y(d) über der Tiefe d. Eine
    45°-Flanke öffnet sich um 1 mm je mm Tiefe, hat also |slope| = 1.
    """
    return math.degrees(math.atan(abs(slope)))


def extract_features(report: dict) -> dict:
    """Eine Ergebniszeile aus einem Report. Fehlende Blöcke werden zu NaN."""
    dev = report.get("deviation", {})
    out: dict[str, float] = {}

    # ── Ebene 1: global ──────────────────────────────────────────────
    out["reg_residual_mm"] = report.get("registration", {}).get("final_residual", math.nan)
    out["reg_converged"] = report.get("registration", {}).get("converged", False)
    out["global_in_tol_rate"] = dev.get("overall_in_tolerance_rate", math.nan)
    out["n_points"] = dev.get("n_points", math.nan)

    regions = dev.get("per_region_metrics", {})
    # Punktgewichtetes Mittel über die Regionen: der Report führt keinen
    # globalen Kennwert, nur die regionsweisen. Ohne Gewichtung dominierte
    # die kleine Spaltregion den Mittelwert der grossen Oberseite.
    tot_n = sum(r.get("n_valid", 0) for r in regions.values())
    for metric in ("mean_abs", "rms", "mean_signed"):
        if tot_n:
            out[f"global_{metric}"] = sum(
                r.get(metric, 0.0) * r.get("n_valid", 0) for r in regions.values()) / tot_n
        else:
            out[f"global_{metric}"] = math.nan
    out["global_max_abs"] = max((r.get("max_abs", math.nan) for r in regions.values()),
                                default=math.nan)
    for label, name in ((0, "top"), (1, "flank_a"), (2, "flank_b")):
        r = regions.get(str(label), {})
        out[f"region_{name}_mean_abs"] = r.get("mean_abs", math.nan)
        out[f"region_{name}_in_tol"] = r.get("in_tolerance_rate", math.nan)

    # ── Ebene 2: lokal (Voxel) ───────────────────────────────────────
    vox = dev.get("voxel_deviation", {})
    v_mean_abs = _finite(vox.get("mean_abs", []))
    out["vox_size_mm"] = vox.get("voxel_size_mm", math.nan)
    out["vox_n"] = vox.get("n_voxels", 0)
    out["vox_max_mean_abs"] = float(v_mean_abs.max()) if v_mean_abs.size else math.nan
    out["vox_frac_out_of_tol"] = (float((v_mean_abs > TOLERANCE_MM).mean())
                                  if v_mean_abs.size else math.nan)
    v_in_tol = _finite(vox.get("in_tolerance_rate", []))
    out["vox_worst_in_tol"] = float(v_in_tol.min()) if v_in_tol.size else math.nan

    # ── Ebene 3: Merkmale ────────────────────────────────────────────
    comp = dev.get("component_registration") or {}
    tr, rot = comp.get("translation_mm", {}), comp.get("rotation_deg", {})
    for axis in "xyz":
        out[f"measured_t{axis}_mm"] = tr.get(axis, math.nan)
        out[f"measured_r{axis}_deg"] = rot.get(axis, math.nan)
    out["residual_a_mm"] = comp.get("residual_a_mm", math.nan)
    out["residual_b_mm"] = comp.get("residual_b_mm", math.nan)

    gap = dev.get("gap_profile") or {}
    out["anchored"] = gap.get("anchored", False)

    fa, fb = gap.get("flank_a_profile", {}), gap.get("flank_b_profile", {})
    out["flank_a_angle_deg"] = slope_to_angle_deg(_nanmean(fa.get("slope", [])))
    out["flank_b_angle_deg"] = slope_to_angle_deg(_nanmean(fb.get("slope", [])))
    out["flank_asymmetry_deg"] = abs(out["flank_a_angle_deg"] - out["flank_b_angle_deg"])
    out["flank_a_r2"] = _nanmean(fa.get("r2", []))
    out["flank_b_r2"] = _nanmean(fb.get("r2", []))
    out["flank_a_rms_mm"] = _nanmean(fa.get("rms", []))
    out["flank_b_rms_mm"] = _nanmean(fb.get("rms", []))
    out["fit_quality_min_r2"] = min(out["flank_a_r2"], out["flank_b_r2"])

    ovr = gap.get("opposite_vs_reference", {})
    out["edge_offset_mm"] = ovr.get("height_offset_mm", math.nan)
    out["tilt_total_deg"] = ovr.get("tilt_total_deg", math.nan)
    out["tilt_along_seam_deg"] = ovr.get("tilt_along_seam_deg", math.nan)
    out["tilt_across_gap_deg"] = ovr.get("tilt_across_gap_deg", math.nan)

    ref, opp = gap.get("reference_plane", {}), gap.get("opposite_plane", {})
    out["ref_plane_inlier_ratio"] = ref.get("inlier_ratio", math.nan)
    out["ref_plane_rms_mm"] = ref.get("rms_mm", math.nan)
    out["opp_plane_inlier_ratio"] = opp.get("inlier_ratio", math.nan)

    raw_widths = np.array(
        [np.nan if v is None else v for v in gap.get("gap_root_widths", [])], dtype=float)
    centers = np.array(gap.get("seam_axis_centers", []), dtype=float)
    widths, d_roots = _finite(gap.get("gap_root_widths", [])), _finite(gap.get("d_root", []))
    n_bins = len(centers)
    out["n_bins"] = n_bins
    out["n_bins_valid"] = int(widths.size)
    out["bin_coverage"] = widths.size / n_bins if n_bins else math.nan
    out["gap_width_mm"] = float(widths.mean()) if widths.size else math.nan
    out["gap_width_std_mm"] = float(widths.std()) if widths.size else math.nan
    out["d_root_mm"] = float(d_roots.mean()) if d_roots.size else math.nan

    # Verkippt ein Werkstück um die Hochachse, oeffnet sich der Spalt keilfoermig
    # entlang der Naht. Dann ist die mittlere Breite kein sinnvoller Kennwert
    # mehr, die Steigung ueber der Nahtlaenge dagegen schon.
    ok = np.isfinite(raw_widths)
    if ok.sum() >= 3:
        slope, _ = np.polyfit(centers[ok], raw_widths[ok], 1)
        out["gap_wedge_slope"] = float(slope)
    else:
        out["gap_wedge_slope"] = math.nan
    return out


def expected_gap_width_mm(d_root_mm: float, ty_mm: float, tz_mm: float = 0.0) -> float:
    """Sollbreite auf der tatsächlich ausgewerteten Tiefe.

    Drei Beiträge, alle geometrisch und keiner davon ein Messfehler:

    * **Resttiefe** – Der Wurzelspalt wird nicht an der Wurzel gemessen,
      sondern auf der tiefsten beidseitig belegten Tiefe `d_root`. Weil sich
      die V-Naht nach oben öffnet, ist die dort erwartete Breite grösser als
      das Nennmass, und zwar um `2·tan(α)` je mm Resttiefe. Ohne diesen Term
      wäre der Messwert scheinbar um rund 0.5 mm zu gross.
    * **ty** – Eine Verschiebung quer zum Spalt öffnet ihn unmittelbar.
    * **tz** – Hebt sich ein Werkstück, so liegt auf einer festen Tiefe unter
      der Referenzebene nun tieferes und damit engeres Flankenmaterial. Der
      Spalt wird um `tan(α)·tz` schmaler, bei 45° also um `tz`.

    Nicht enthalten sind `ry` und `rz`: beide erzeugen einen entlang der Naht
    *veränderlichen* Spalt. Dort ist die mittlere Breite kein sinnvoller
    Vergleichswert; herangezogen wird stattdessen `gap_wedge_slope`.
    """
    tan_a = math.tan(math.radians(NOMINAL_FLANK_ANGLE_DEG))
    residual_depth = NOMINAL_THICKNESS_MM - d_root_mm
    return (NOMINAL_ROOT_GAP_MM + ty_mm - tan_a * tz_mm
            + 2.0 * residual_depth * tan_a)


def load_results(outputs_dir: Path, metadata_csv: Path) -> pd.DataFrame:
    """Ground Truth und Messung zu einer Tabelle verbinden."""
    gt = pd.read_csv(metadata_csv)
    rows, missing = [], []
    for _, r in gt.iterrows():
        case = str(r["filename"]).replace(".ply", "")
        path = outputs_dir / case / "subtraction_report.json"
        if not path.exists():
            missing.append(case)
            continue
        with open(path, encoding="utf-8") as fh:
            report = json.load(fh)
        rows.append({**r.to_dict(), "case_name": case, **extract_features(report)})

    df = pd.DataFrame(rows)
    df.attrs["missing"] = missing
    if df.empty:
        return df

    df["error_class"] = df["category"].map(ERROR_CLASS)
    for col, sign in SIGN.items():
        df[f"expected_{col}"] = sign * df[col]
        measured = f"measured_{col}"
        if measured in df:
            df[f"err_{col}"] = df[measured] - df[f"expected_{col}"]

    df["gt_translation_norm"] = np.sqrt(df.tx_mm**2 + df.ty_mm**2 + df.tz_mm**2)
    df["gt_rotation_norm"] = np.sqrt(df.rx_deg**2 + df.ry_deg**2 + df.rz_deg**2)

    df["expected_gap_width_mm"] = [
        expected_gap_width_mm(d, ty, tz)
        for d, ty, tz in zip(df.d_root_mm, df.ty_mm, df.tz_mm)]
    df["gap_residual_mm"] = df.gap_width_mm - df.expected_gap_width_mm

    # Bei Verkippung um die Hochachse oder die Spaltachse variiert der Spalt
    # entlang der Naht. Für diese Fälle ist die mittlere Breite kein
    # aussagekräftiger Vergleich – sie werden getrennt über die Keilsteigung
    # geprüft.
    df["gap_model_complete"] = (df.ry_deg == 0) & (df.rz_deg == 0)
    df["expected_wedge_slope"] = -np.tan(np.radians(df.rz_deg))
    df["wedge_slope_err"] = df.gap_wedge_slope - df.expected_wedge_slope

    # Kantenversatz: der Generator verstellt B in Z, gemessen wird der
    # Höhenversatz der Gegenseite gegen die Referenzebene.
    df["expected_edge_offset_mm"] = df.tz_mm
    df["edge_offset_err_mm"] = df.edge_offset_mm - df.expected_edge_offset_mm

    df["flank_a_angle_err_deg"] = df.flank_a_angle_deg - NOMINAL_FLANK_ANGLE_DEG
    df["flank_b_angle_err_deg"] = df.flank_b_angle_deg - NOMINAL_FLANK_ANGLE_DEG
    return df
