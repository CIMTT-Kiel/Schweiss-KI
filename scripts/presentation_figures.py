#!/usr/bin/env python3
"""Statische Präsentationsbilder aus dem synthetischen Datensatz.

Für ein Team mit grobem Projektwissen gedacht — die Bilder müssen ohne
Methodik-Erklärung lesbar sein. Drei Bilder:

  1. Abweichungs-Farbkarte: CAD-Ideal gegen drei Scans, eingefärbt nach
     signiertem Abstand (Material fehlt / in Toleranz / steht über)
  2. C_TR_08-Dreiklang: warum es drei Auswertungsebenen braucht
  3. Vorher-Nachher der Verankerung: Anteil in Toleranz und RMS

Aufruf:  uv run python scripts/presentation_figures.py
Ausgabe: docs/figures/praesentation/*.png (dpi=200)

Statischer Renderer bewusst Matplotlib: Plotly braucht für PNG-Export kaleido
(Chromium-Subprozess, läuft hier nicht), Open3D-Offscreen scheitert an EGL. Das
interaktive Dashboard entsteht separat mit Plotly-HTML.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from schweiss_ki.core.console import force_utf8_output
from schweiss_ki.analysis.deviation_field import (
    TOLERANCE_MM, load_cad_reference, signed_distance_field, subsample,
)

OUTPUTS = ROOT / "data" / "outputs"
FIGDIR = ROOT / "docs" / "figures" / "praesentation"
DPI = 200

# Diverging-Paar aus der Design-Referenz: blau (fehlt) ↔ neutralgrau (in
# Toleranz) ↔ rot (steht über). Poles lesen als Gegensatz, Mitte als "nichts".
C_MISSING = "#2a78d6"
C_NEUTRAL = "#e8e7e2"   # etwas kräftiger als die reine Mitte, damit ein
                        # rundum tolerantes Teil nicht auf Weiß verschwindet
C_EXCESS = "#e34948"
DEV_CMAP = LinearSegmentedColormap.from_list(
    "abweichung", [C_MISSING, "#a9c6ea", C_NEUTRAL, "#eeab9f", C_EXCESS])

# Status (fest, nie umgefärbt) fürs Vorher-Nachher
C_BAD = "#d03b3b"
C_GOOD = "#0ca30c"

# Ink-Token
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
C_CAD = "#c3c2b7"

VMAX = 2.0   # gemeinsame, symmetrische Farbgrenze über alle Fälle (mm)

CASES = [
    ("T_X_+00.100mm", "Sauberes Teil", "0.1 mm längs verschoben"),
    ("R_X_+02.000deg", "Verkipptes Teil", "ein Werkstück um 2° gekippt"),
    ("C_TR_08", "Deutliche Fehlstellung", "1 mm Höhenversatz + 1° Verkippung"),
]
FOOTER = "Synthetische Daten"


def _plate_axes(ax):
    ax.set_aspect("equal")
    ax.set_xlabel("Naht-Längsrichtung (mm)", fontsize=9, color=INK2)
    ax.set_ylabel("quer (mm)", fontsize=9, color=INK2)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def _footer(fig):
    fig.text(0.5, 0.012, FOOTER, ha="center", fontsize=8.5, color=MUTED,
             style="italic")


# ── Bild 1: Abweichungs-Farbkarte ─────────────────────────────────────

def figure_deviation_map(cad, path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.4), dpi=DPI)
    fig.subplots_adjust(left=0.06, right=0.83, top=0.87, bottom=0.11,
                        wspace=0.18, hspace=0.52)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-VMAX, vmax=VMAX)

    def _panel_title(ax, main, sub=None):
        ax.set_title(main, fontsize=11, color=INK, weight="bold", pad=24)
        if sub:
            ax.text(0.5, 1.03, sub, transform=ax.transAxes, ha="center",
                    va="bottom", fontsize=8.5, color=MUTED)

    # (0,0) CAD-Ideal als Soll-Silhouette (Oberseite, ein Ton)
    ax = axes[0, 0]
    top = cad.points[cad.top_mask]
    sel = subsample(len(top), 30000, seed=1)
    ax.scatter(top[sel, 0], top[sel, 1], s=2, c=C_CAD, marker="s",
               linewidths=0, rasterized=True)
    _panel_title(ax, "CAD-Ideal (So soll es sein)",
                 "durchgehender Spalt, keine Abweichung")
    _plate_axes(ax)

    # drei Scans, gemeinsame Farbskala
    mappable = None
    for ax, (case, title, sub) in zip(axes.flat[1:], CASES):
        field = signed_distance_field(OUTPUTS / case, cad)
        sel = subsample(len(field.points), 45000, seed=2)
        p, d = field.points[sel], field.signed[sel]
        # nach |d| sortiert: die Abweichungen kommen zuoberst zu liegen
        order = np.argsort(np.abs(d))
        mappable = ax.scatter(p[order, 0], p[order, 1], c=d[order], cmap=DEV_CMAP,
                              norm=norm, s=2, marker="s", linewidths=0,
                              rasterized=True)
        rate = field.in_tolerance_rate * 100
        _panel_title(ax, f"{title} — {rate:.0f} % in Toleranz", sub)
        _plate_axes(ax)

    fig.suptitle("Abweichung zum CAD-Ideal — Scan eingefärbt nach Abstand",
                 fontsize=15, weight="bold", color=INK, y=0.965)

    # gemeinsame Farbleiste rechts, eigenes Achsenobjekt (kollidiert nicht)
    cax = fig.add_axes([0.865, 0.16, 0.016, 0.56])
    # Kein Achsentitel: die Laien-Endlabels (fehlt / in Toleranz / steht über)
    # erklären die Skala vollständig, "signierter Abstand" wäre Fachjargon und
    # kreuzte die Beschriftung.
    cbar = fig.colorbar(mappable, cax=cax, extend="both")
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.ax.axhspan(-TOLERANCE_MM, TOLERANCE_MM, facecolor="none",
                    edgecolor=INK, linewidth=1.0)
    for y, txt, va in ((VMAX, "Material\nsteht über", "top"),
                       (0.0, "in Toleranz\n(±0.25 mm)", "center"),
                       (-VMAX, "Material\nfehlt", "bottom")):
        cbar.ax.text(2.6, y, txt, transform=cbar.ax.get_yaxis_transform(),
                     ha="left", va=va, fontsize=8.5, color=INK2)

    _footer(fig)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── Bild 2: C_TR_08-Dreiklang ─────────────────────────────────────────

def _load_report(case: str) -> dict:
    with open(OUTPUTS / case / "subtraction_report.json", encoding="utf-8") as fh:
        return json.load(fh)


def figure_three_levels(cad, path: Path):
    case = "C_TR_08"
    rep = _load_report(case)["deviation"]
    field = signed_distance_field(OUTPUTS / case, cad)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.0), dpi=DPI,
                             gridspec_kw={"width_ratios": [1, 1.5, 1.15]})

    # Ebene 1 — ein Wert
    ax = axes[0]
    ax.axis("off")
    rate = field.in_tolerance_rate * 100
    ax.text(0.5, 0.72, f"{rate:.0f} %", ha="center", va="center",
            fontsize=64, color=C_BAD, weight="bold")
    ax.text(0.5, 0.50, "der Punkte in Toleranz", ha="center", fontsize=12,
            color=INK)
    # schlichter Fortschrittsbalken 0..100
    ax.add_patch(plt.Rectangle((0.15, 0.34), 0.70, 0.05, color=GRID,
                               transform=ax.transAxes))
    ax.add_patch(plt.Rectangle((0.15, 0.34), 0.70 * rate / 100, 0.05,
                               color=C_BAD, transform=ax.transAxes))
    ax.text(0.5, 0.16, "Ein Wert fürs ganze Teil.\nSagt: schlecht.\n"
            "Sagt nicht: wo.", ha="center", va="top", fontsize=11, color=INK2)
    ax.set_title("① Global", fontsize=13, color=INK, weight="bold", loc="left")

    # Ebene 2 — Voxel zeigt WO
    ax = axes[1]
    vox = rep["voxel_deviation"]
    centers = np.array(vox["centers"])
    mean_signed = np.array(vox["mean_signed"], dtype=float)
    vmax = float(np.nanpercentile(np.abs(mean_signed), 98))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    m = ax.scatter(centers[:, 0], centers[:, 1], c=mean_signed, cmap=DEV_CMAP,
                   norm=norm, s=30, marker="s", linewidths=0)
    cbar = fig.colorbar(m, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Abstand je Würfel (mm)", fontsize=8, color=INK2)
    cbar.ax.tick_params(colors=MUTED, labelsize=7)
    ax.set_title("② Voxel — zeigt WO", fontsize=13, color=INK, weight="bold",
                 loc="left")
    ax.text(0.5, -0.30, "Die Abweichung sitzt nicht überall gleich —\n"
            "sie ballt sich in einer Ecke.", transform=ax.transAxes,
            ha="center", fontsize=9.5, color=MUTED)
    _plate_axes(ax)
    ax.set_ylabel("quer (mm)", fontsize=9, color=INK2)

    # Ebene 3 — Merkmale benennen WAS (Stat-Liste, gemischte Einheiten)
    ax = axes[2]
    ax.axis("off")
    ax.set_title("③ Merkmale — benennen WAS", fontsize=13, color=INK,
                 weight="bold", loc="left")
    gp = rep["gap_profile"]["opposite_vs_reference"]
    fa = np.array(rep["gap_profile"]["flank_a_profile"]["slope"], dtype=float)
    fb = np.array(rep["gap_profile"]["flank_b_profile"]["slope"], dtype=float)
    asym = abs(np.degrees(np.arctan(np.abs(np.nanmean(fa))))
               - np.degrees(np.arctan(np.abs(np.nanmean(fb)))))
    rows = [
        (f"{gp['height_offset_mm']:.2f} mm", "Kantenversatz",
         "Bauteile höhenversetzt"),
        (f"{gp['tilt_total_deg']:.2f}°", "Verkippung",
         "ein Teil gegen das andere gekippt"),
        (f"{asym:.2f}°", "Flankenasymmetrie", "Flanken ungleich steil"),
    ]
    y = 0.74
    for value, name, desc in rows:
        ax.text(0.04, y, value, fontsize=22, color=INK, weight="bold",
                va="center")
        ax.text(0.04, y - 0.09, name, fontsize=11, color=INK, va="center")
        ax.text(0.04, y - 0.155, desc, fontsize=9, color=MUTED, va="center")
        y -= 0.30

    fig.suptitle("Warum drei Auswertungsebenen — am Fall C_TR_08",
                 fontsize=14, weight="bold", color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    _footer(fig)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── Bild 3: Vorher-Nachher ────────────────────────────────────────────

def figure_before_after(path: Path):
    # Werte aus fehleranalyse_achsen_und_registrierung.md
    in_tol = (63.9, 100.0)
    rms = (0.4154, 0.0017)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), dpi=DPI)
    labels = ["vor der\nVerankerung", "nach der\nVerankerung"]
    colors = [C_BAD, C_GOOD]

    ax = axes[0]
    bars = ax.bar(labels, in_tol, color=colors, width=0.6, zorder=3)
    for b, v in zip(bars, in_tol):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f} %",
                ha="center", fontsize=13, color=INK, weight="bold")
    ax.set_ylim(0, 112)
    ax.set_ylabel("Anteil der Naht in Toleranz (%)", fontsize=10, color=INK2)
    ax.set_title("Wie viel liegt in Toleranz?", fontsize=12, color=INK,
                 weight="bold")

    ax = axes[1]
    bars = ax.bar(labels, rms, color=colors, width=0.6, zorder=3)
    for b, v in zip(bars, rms):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.4f} mm",
                ha="center", fontsize=13, color=INK, weight="bold")
    ax.set_ylim(0, 0.47)
    ax.set_ylabel("Registrierungs-Einfluss RMS (mm)", fontsize=10, color=INK2)
    ax.set_title("Wie stark verfälscht die Ausrichtung?", fontsize=12,
                 color=INK, weight="bold")
    ax.annotate("240× kleiner", xy=(1, rms[1]), xytext=(1, 0.22),
                ha="center", fontsize=11, color=C_GOOD, weight="bold",
                arrowprops=dict(arrowstyle="->", color=C_GOOD, lw=1.6))

    for ax in axes:
        ax.tick_params(colors=MUTED, labelsize=10)
        ax.tick_params(axis="x", colors=INK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)

    fig.suptitle("Was die Verankerung gebracht hat", fontsize=14,
                 weight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    _footer(fig)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    force_utf8_output()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    cad = load_cad_reference(OUTPUTS)
    figure_deviation_map(cad, FIGDIR / "01_abweichungskarte.png")
    figure_three_levels(cad, FIGDIR / "02_drei_ebenen.png")
    figure_before_after(FIGDIR / "03_vorher_nachher.png")
    print(f"Drei Bilder geschrieben nach {FIGDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
