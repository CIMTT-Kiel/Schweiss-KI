"""AP2.4 — Füllvolumen der Schweißnaht aus der Nutgeometrie.

Wie viel Schweißgut in die Naht eingebracht werden muss, ergibt sich rein
geometrisch aus dem, was das Gap-Profil ohnehin liefert: pro Bin beide Flanken
als Gerade y(d) = q0 + slope·d über der Tiefe d unter der Oberseite. Die
Spaltbreite ist w(d) = y_B(d) − y_A(d), die Nut-Querschnittsfläche das Integral
über die Materialstärke T:

    A = ∫₀ᵀ w(d) dd = T · (w(0) + w(T)) / 2

weil w(d) linear ist — exakt die Trapezform der V-Naht. Das Füllvolumen ist die
Fläche über die Nahtlänge integriert. „Öffnungswinkel, Spaltbreite und
Materialstärke" aus dem Arbeitsplan stecken genau hier: Winkel → slope,
Spaltbreite → w, Dicke → T.

Nahtlänge: die volle beidseitig belegte Nut (`seam_length_mm` im Gap-Profil),
nicht der um den `edge_margin` gekürzte Fit-Bereich. Der Margin hält nur die
Heftpunkte aus den Flankenfits heraus; die Nut ist physisch durchgehend. Die
mittlere Querschnittsfläche wird deshalb über die volle Nutlänge integriert.

Materialstärke T ist ein Eingabewert (Bauteilvorgabe), kein Messwert — der Scan
von oben sieht die Unterseite nicht. Default 5.0 mm für das Referenzbauteil.

Nahtüberhöhung: optionaler Aufschlag für die über die Oberseite stehende Kappe.
Als parabolische Raupe über der Öffnungsbreite modelliert, A_kappe = 2/3·w_oben·h
(Standardnäherung für die Nahtüberhöhung). Default 0 — reine Nutfüllung.

FEM-Alternative aus dem Arbeitsplan: für die prismatische V-Naht nicht nötig,
der geometrische Weg ist exakt. Erst bei komplexeren Nahtformen relevant.
"""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

DEFAULT_THICKNESS_MM = 5.0


def _arr(seq) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in seq], dtype=float)


def cross_section_areas(report: Dict[str, Any],
                        thickness_mm: float = DEFAULT_THICKNESS_MM,
                        reinforcement_mm: float = 0.0) -> np.ndarray:
    """Nut-Querschnittsfläche je Bin in mm². NaN, wo eine Flanke fehlt.

    reinforcement_mm > 0 addiert die Nahtüberhöhung (parabolische Kappe über der
    Öffnungsbreite, A = 2/3·w_oben·h) auf jede Bin-Fläche.
    """
    gp = report["deviation"].get("gap_profile") or {}
    fa, fb = gp.get("flank_a_profile", {}), gp.get("flank_b_profile", {})
    if not fa or not fb:
        return np.array([])
    q0a, sa = _arr(fa["q0"]), _arr(fa["slope"])
    q0b, sb = _arr(fb["q0"]), _arr(fb["slope"])
    t = float(thickness_mm)
    w_top = q0b - q0a                         # Spalt an der Oberseite (d=0)
    w_root = (q0b + sb * t) - (q0a + sa * t)  # Spalt an der Wurzel (d=T)
    area = t * (w_top + w_root) / 2.0
    h = float(reinforcement_mm)
    if h > 0.0:
        area = area + (2.0 / 3.0) * w_top * h
    return area


def fill_volume(report: Dict[str, Any],
                thickness_mm: float = DEFAULT_THICKNESS_MM,
                reinforcement_mm: float = 0.0) -> Dict[str, Any]:
    """Füllvolumen der Naht plus die Verteilung entlang der Naht.

    Rückgabe:
        fill_volume_mm3    – Gesamt-Füllvolumen (mittlere Fläche × Nahtlänge)
        mean_area_mm2      – mittlere Querschnittsfläche
        area_min/max_mm2   – Spanne der Fläche (zeigt Keil/Ungleichmäßigkeit)
        seam_length_mm     – volle Nutlänge (Basis des Volumens)
        measured_length_mm – tatsächlich befittete Länge (ohne edge_margin)
        thickness_mm       – verwendete Materialstärke
        reinforcement_mm   – verwendete Nahtüberhöhung
        n_bins_valid       – Bins mit beidseitiger Flanke
        seam_positions_mm  – Bin-Mitten (für die Verteilung)
        area_profile_mm2   – Querschnittsfläche je Bin (für Roboterbahn/Rate)
    """
    gp = report["deviation"].get("gap_profile") or {}
    centers = _arr(gp.get("seam_axis_centers", []))
    areas = cross_section_areas(report, thickness_mm, reinforcement_mm)
    if areas.size == 0 or centers.size == 0:
        return {"fill_volume_mm3": math.nan, "n_bins_valid": 0}

    valid = np.isfinite(areas)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return {"fill_volume_mm3": math.nan, "n_bins_valid": 0}

    # Volle Nutlänge aus dem Report (vor edge_margin). Fallback für ältere
    # Reports: Bin-Spanne (unterschätzt um 2·edge_margin).
    seam_length = gp.get("seam_length_mm")
    if seam_length is None:
        dx = float(np.mean(np.diff(centers))) if centers.size > 1 else 0.0
        seam_length = abs(dx) * len(centers)
    seam_length = float(seam_length)

    dx = float(np.mean(np.diff(centers))) if centers.size > 1 else 0.0
    measured_length = abs(dx) * len(centers)
    mean_area = float(np.nanmean(areas[valid]))

    return {
        "fill_volume_mm3": mean_area * seam_length,
        "mean_area_mm2": mean_area,
        "area_min_mm2": float(np.nanmin(areas[valid])),
        "area_max_mm2": float(np.nanmax(areas[valid])),
        "seam_length_mm": seam_length,
        "measured_length_mm": measured_length,
        "thickness_mm": float(thickness_mm),
        "reinforcement_mm": float(reinforcement_mm),
        "n_bins_valid": n_valid,
        "n_bins": int(centers.size),
        "seam_positions_mm": centers.tolist(),
        "area_profile_mm2": areas.tolist(),
    }
