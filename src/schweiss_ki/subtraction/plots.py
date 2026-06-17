"""
Plot-Funktionen für Subtraktions-Ergebnisse.

Bereitstellt wiederverwendbare Matplotlib-Plots, die sowohl von der
Pipeline (automatisch nach _run_subtraction) als auch von Notebooks
(interaktive Validierung) genutzt werden können.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from ..core.data_structures import WeldVolumeModel

logger = logging.getLogger(__name__)


# ── Farb-Konvention (konsistent mit Notebook-Plots) ────────────────────
COLOR_MEASURED = "#1976D2"   # blau – gemessener Verlauf
COLOR_SOLL = "#37474F"       # dunkelgrau – Soll-Linie
COLOR_TOLERANCE = "#B0BEC5"  # hellgrau – Toleranzband


def plot_gap_profile(
    model: "WeldVolumeModel",
    gap_soll_mm: Optional[float] = None,
    title_suffix: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plottet den Wurzelspalt-Verlauf entlang der Naht-Längsrichtung.

    Args:
        model: WeldVolumeModel mit gesetztem subtraction_report und gap_profile.
        gap_soll_mm: Optional Soll-Spaltbreite. Wenn None: kein Soll/Toleranzband
            im Plot (z.B. wenn Soll-Wert unbekannt).
        title_suffix: Optionaler Zusatz im Plot-Titel (z.B. Bauteil-Beschreibung).

    Returns:
        Matplotlib-Figure, oder None falls keine Gap-Profil-Daten vorhanden.
    """
    if not model.has_subtraction:
        logger.warning(
            f"  plot_gap_profile: model '{model.model_id}' hat keinen "
            f"subtraction_report – Plot übersprungen."
        )
        return None

    gp = model.subtraction_report.deviation.gap_profile
    if gp is None:
        logger.warning(
            f"  plot_gap_profile: model '{model.model_id}' hat keine "
            f"gap_profile-Daten – Plot übersprungen."
        )
        return None

    x_centers = np.asarray(gp["seam_axis_centers"])
    gap_widths = np.asarray(gp["gap_widths"])
    tolerance_mm = model.subtraction_report.deviation.tolerance_mm

    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)

    # Toleranzband + Soll-Linie (nur wenn gap_soll_mm gesetzt)
    if gap_soll_mm is not None:
        ax.axhspan(
            gap_soll_mm - tolerance_mm,
            gap_soll_mm + tolerance_mm,
            color=COLOR_TOLERANCE, alpha=0.3,
            label=f"Toleranz ±{tolerance_mm} mm",
        )
        ax.axhline(
            gap_soll_mm,
            color=COLOR_SOLL, linestyle="--", linewidth=1.8,
            label=f"Soll ({gap_soll_mm:.1f} mm konstant)",
        )

    # Gemessener Verlauf
    ax.plot(
        x_centers, gap_widths,
        color=COLOR_MEASURED, linewidth=1.8, marker="o", markersize=4,
        label="Gemessen (CMM-Scan)",
    )

    ax.set_xlabel("Position entlang Schweißnaht (mm)")
    ax.set_ylabel("Spaltbreite (mm)")

    title = f"Spaltbreitenverlauf entlang der V-Naht – {model.model_id}"
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
    """Erstellt den Spaltprofil-Plot und speichert ihn als PNG.

    Args:
        model: WeldVolumeModel mit gap_profile-Daten.
        output_path: Zielpfad für die PNG-Datei.
        gap_soll_mm: Optional Soll-Spaltbreite.
        title_suffix: Optionaler Zusatz im Titel.
        dpi: Auflösung.

    Returns:
        Pfad zur gespeicherten Datei, oder None falls Plot nicht erzeugt werden konnte.
    """
    fig = plot_gap_profile(
        model,
        gap_soll_mm=gap_soll_mm,
        title_suffix=title_suffix,
    )
    if fig is None:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.debug(f"  Spaltprofil-Plot gespeichert: {output_path}")
    return output_path