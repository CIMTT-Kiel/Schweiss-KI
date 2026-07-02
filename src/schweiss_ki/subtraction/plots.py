"""
Plot-Funktionen für Subtraktions-Ergebnisse und Diagnose.

Bereitstellt wiederverwendbare Matplotlib-Plots, die sowohl von der
Pipeline (automatisch nach _run_subtraction) als auch von Notebooks
(interaktive Validierung) genutzt werden können.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

if TYPE_CHECKING:
    from ..core.data_structures import WeldVolumeModel

logger = logging.getLogger(__name__)


# ── Farb-Konvention (konsistent mit Notebook-Plots) ────────────────────
COLOR_MEASURED = "#1976D2"   # blau – gemessener Verlauf
COLOR_SOLL = "#37474F"       # dunkelgrau – Soll-Linie
COLOR_TOLERANCE = "#B0BEC5"  # hellgrau – Toleranzband
COLOR_CAD = "#1976D2"        # blau für CAD im Overlay
COLOR_SCAN = "#E53935"       # rot für Scan im Overlay

LABEL_NAMES = {
    0: "Background",
    1: "Flanke A",
    2: "Flanke B",
    3: "Spalt",
    4: "Sub-Gap",
}

LABEL_COLORS = {
    0: "#B0BEC5",   # hellgrau
    1: "#1976D2",   # blau
    2: "#FF9800",   # orange
    3: "#43A047",   # grün
    4: "#9C27B0",   # lila
}


# ── Spaltprofil ────────────────────────────────────────────────────────

def plot_gap_profile(
    model: "WeldVolumeModel",
    gap_soll_mm: Optional[float] = None,
    title_suffix: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plottet den Wurzelspalt-Verlauf entlang der Naht-Längsrichtung.

    Args:
        model: WeldVolumeModel mit gesetztem subtraction_report und gap_profile.
        gap_soll_mm: Optional Soll-Spaltbreite. Wenn None: kein Soll/Toleranzband.
        title_suffix: Optionaler Zusatz im Plot-Titel.

    Returns:
        Matplotlib-Figure oder None falls keine Gap-Profil-Daten vorhanden.
    """
    if not model.has_subtraction:
        logger.warning(f"  plot_gap_profile: model '{model.model_id}' ohne subtraction_report")
        return None

    gp = _get_gap_profile_data(model)
    if gp is None:
        logger.warning(f"  plot_gap_profile: model '{model.model_id}' ohne gap_profile-Daten")
        return None

    x_centers = np.asarray(gp["seam_axis_centers"])
    gap_widths = np.asarray(gp["gap_widths"])
    tolerance_mm = _get_tolerance(model)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)

    if gap_soll_mm is not None:
        ax.axhspan(
            gap_soll_mm - tolerance_mm, gap_soll_mm + tolerance_mm,
            color=COLOR_TOLERANCE, alpha=0.3,
            label=f"Toleranz ±{tolerance_mm} mm",
        )
        ax.axhline(
            gap_soll_mm,
            color=COLOR_SOLL, linestyle="--", linewidth=1.8,
            label=f"Soll ({gap_soll_mm:.1f} mm konstant)",
        )

    ax.plot(
        x_centers, gap_widths,
        color=COLOR_MEASURED, linewidth=1.8, marker="o", markersize=4,
        label="Gemessen (CMM-Scan)",
    )
    ax.set_xlabel("Position entlang Schweißnaht (mm)")
    ax.set_ylabel("Spaltbreite (mm)")
    title = f"Spaltbreitenverlauf – {model.model_id}"
    if title_suffix:
        title = f"{title}\n{title_suffix}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    return fig


def save_gap_profile_png(
    model: "WeldVolumeModel",
    output_path: Path,
    gap_soll_mm: Optional[float] = None,
    title_suffix: Optional[str] = None,
    dpi: int = 200,
) -> Optional[Path]:
    """Erstellt den Spaltprofil-Plot und speichert ihn als PNG."""
    fig = plot_gap_profile(model, gap_soll_mm=gap_soll_mm, title_suffix=title_suffix)
    if fig is None:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.debug(f"  Spaltprofil-Plot gespeichert: {output_path}")
    return output_path


# ── Segmentierungs-Übersicht ───────────────────────────────────────────

def plot_segmentation_overview(
    model: "WeldVolumeModel",
    view: str = "top",
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Draufsicht (XY) oder Seitenansicht (XZ) mit Label-Farben.

    Args:
        model: WeldVolumeModel mit Labels.
        view: 'top' (XY), 'side' (XZ) oder 'front' (YZ).
        figsize: Figure-Größe.
    """
    pts = np.asarray(model.point_cloud.points)
    if model.labels is None:
        raise ValueError(f"Model '{model.model_id}' hat keine Labels.")

    axis_map = {"top": (0, 1, "X (mm)", "Y (mm)"),
                "side": (0, 2, "X (mm)", "Z (mm)"),
                "front": (1, 2, "Y (mm)", "Z (mm)")}
    if view not in axis_map:
        raise ValueError(f"view muss 'top', 'side' oder 'front' sein, war '{view}'.")
    x_idx, y_idx, xlabel, ylabel = axis_map[view]

    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    for label_id, label_name in LABEL_NAMES.items():
        mask = model.labels == label_id
        if not mask.any():
            continue
        ax.scatter(
            pts[mask, x_idx], pts[mask, y_idx],
            c=LABEL_COLORS[label_id], s=2, alpha=0.6,
            label=f"{label_id}: {label_name} ({int(mask.sum()):,})",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    ax.set_title(f"Segmentierung [{view}] – {model.model_id}")
    ax.legend(markerscale=5, framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ── Querschnitt für Diagnose ───────────────────────────────────────────

def plot_cross_section(
    model: "WeldVolumeModel",
    x_min: float,
    x_max: float,
    show_extrapolation: bool = True,
    labels_to_show: Sequence[int] = (1, 2, 3, 4),
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Querschnitt YZ für einen X-Bereich entlang der Naht.

    Zeigt die Punkte aller relevanten Labels mit ihrer Z-Höhe gegen die
    Y-Position. Wenn show_extrapolation=True: zeichnet die linearen Fits
    durch Flanke A und B ein, plus deren Schnittpunkt mit z=0 (der
    virtuelle Wurzelpunkt, der für die Spaltbreitenmessung verwendet wird).

    Diagnostisches Werkzeug: Gratbildung, Wurzeldurchhang, ungewöhnliche
    Punktverteilungen sind hier sofort sichtbar.
    """
    pts = np.asarray(model.point_cloud.points)
    if model.labels is None:
        raise ValueError(f"Model '{model.model_id}' hat keine Labels.")

    mask = (pts[:, 0] >= x_min) & (pts[:, 0] < x_max)
    if not mask.any():
        raise ValueError(f"Keine Punkte im X-Bereich [{x_min}, {x_max}].")

    fig, ax = plt.subplots(figsize=figsize, dpi=200)

    # Punkte je Label
    for label_id in labels_to_show:
        lmask = mask & (model.labels == label_id)
        if not lmask.any():
            continue
        ax.scatter(
            pts[lmask, 1], pts[lmask, 2],
            c=LABEL_COLORS[label_id], s=15, alpha=0.7,
            label=f"{LABEL_NAMES[label_id]} ({int(lmask.sum())})",
        )

    # Extrapolations-Linien für Flanke A und B
    if show_extrapolation:
        for label_id in (1, 2):
            lmask = mask & (model.labels == label_id)
            sub = pts[lmask]
            if len(sub) < 3:
                continue
            z = sub[:, 2]
            y = sub[:, 1]
            slope, y0 = np.polyfit(z, y, 1)
            z_line = np.linspace(min(z.min(), -0.5), z.max(), 50)
            y_line = slope * z_line + y0
            ax.plot(
                y_line, z_line, "--",
                color=LABEL_COLORS[label_id], alpha=0.6, linewidth=1.5,
                label=f"{LABEL_NAMES[label_id]} extrapoliert (y₀={y0:+.3f})",
            )
            # Marker am Wurzelpunkt
            ax.scatter([y0], [0], color=LABEL_COLORS[label_id], marker="x", s=80, zorder=10)

    ax.axhline(0, color="gray", linestyle=":", alpha=0.6, label="z = 0 (Wurzelebene)")
    ax.set_xlabel("Y (mm)")
    ax.set_ylabel("Z (mm)")
    ax.set_aspect("equal")
    ax.set_title(f"Querschnitt X ∈ [{x_min:+.1f}, {x_max:+.1f}] – {model.model_id}")
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ── Registrierungs-Overlay ─────────────────────────────────────────────

def plot_registration_overlay(
    model: "WeldVolumeModel",
    cad_pcd: o3d.geometry.PointCloud,
    view: str = "top",
    subsample_n: int = 30_000,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Scan (ausgerichtet) und CAD überlagert.

    Args:
        model: WeldVolumeModel nach Subtraktions-Pipeline.
        cad_pcd: CAD-Punktwolke (z.B. aus data/output/cad/.../pointcloud.ply).
        view: 'top' (XY), 'side' (XZ) oder 'front' (YZ).
        subsample_n: Maximale Punktzahl pro Wolke für Plot-Performance.
    """
    scan_pts = np.asarray(model.point_cloud.points)
    cad_pts = np.asarray(cad_pcd.points)
    rng = np.random.default_rng(0)

    def _sub(pts):
        if len(pts) <= subsample_n:
            return pts
        idx = rng.choice(len(pts), subsample_n, replace=False)
        return pts[idx]

    scan_sub = _sub(scan_pts)
    cad_sub = _sub(cad_pts)

    axis_map = {"top": (0, 1, "X (mm)", "Y (mm)"),
                "side": (0, 2, "X (mm)", "Z (mm)"),
                "front": (1, 2, "Y (mm)", "Z (mm)")}
    if view not in axis_map:
        raise ValueError(f"view muss 'top', 'side' oder 'front' sein, war '{view}'.")
    x_idx, y_idx, xlabel, ylabel = axis_map[view]

    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    ax.scatter(
        cad_sub[:, x_idx], cad_sub[:, y_idx],
        c=COLOR_CAD, s=1, alpha=0.3, label=f"CAD-Ideal ({len(cad_pts):,})",
    )
    ax.scatter(
        scan_sub[:, x_idx], scan_sub[:, y_idx],
        c=COLOR_SCAN, s=2, alpha=0.7, label=f"Scan ausgerichtet ({len(scan_pts):,})",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    ax.set_title(f"Registrierungs-Overlay [{view}] – {model.model_id}")
    ax.legend(markerscale=5, framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ── Helpers ────────────────────────────────────────────────────────────

def _get_gap_profile_data(model: "WeldVolumeModel") -> Optional[dict]:
    """Liest gap_profile sowohl aus subtraction_report.deviation als auch
    aus dem geladenen subtraction_report_raw-Metadata-Dict (load-Fall)."""
    sub = model.subtraction_report
    if sub is None:
        return None

    # 1. Direkter Zugriff (frischer Pipeline-Lauf)
    if hasattr(sub, "deviation") and sub.deviation is not None:
        gp = getattr(sub.deviation, "gap_profile", None)
        if gp:
            return gp

    # 2. Aus den Roh-Daten (geladenes Modell)
    raw = model.metadata.get("subtraction_report_raw")
    if raw and "deviation" in raw:
        gp = raw["deviation"].get("gap_profile")
        if gp:
            return gp
    return None


def _get_tolerance(model: "WeldVolumeModel") -> float:
    """Toleranzwert aus dem Report holen, Default 0.25 mm."""
    sub = model.subtraction_report
    if sub and hasattr(sub, "deviation") and sub.deviation is not None:
        return getattr(sub.deviation, "tolerance_mm", 0.25)
    raw = model.metadata.get("subtraction_report_raw", {})
    return raw.get("deviation", {}).get("tolerance_mm", 0.25)