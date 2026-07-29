#!/usr/bin/env python3
"""Statische Präsentationsbilder aus dem synthetischen Datensatz.

Für ein Team mit grobem Projektwissen gedacht — die Bilder müssen ohne
Methodik-Erklärung lesbar sein. Drei Bilder:

  1. Abweichungs-Farbkarte: CAD-Ideal gegen drei Scans, eingefärbt nach
     signiertem Abstand (Material fehlt / in Toleranz / steht über)
  2. C_TR_08-Dreiklang: warum es drei Auswertungsebenen braucht
  3. Vorher-Nachher der Verankerung: Anteil in Toleranz und RMS

Bild 2 nutzt den eigens erzeugten Fall C_ZX_01 (Drehung um Z + Verschiebung in
X). Der liegt nicht im Standard-Datensatz — vorher einmalig erzeugen:
    uv run python scripts/generate_rztx_demo.py
    uv run python scripts/run_batch_subtraction.py \\
        --scan-dir data/raw/_rztx_demo --source-type synthetic

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
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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

# Achsenkonvention wie im Rest der Doku: X = Naht-Längsrichtung, Y =
# Spalt-Querrichtung, Z = Höhe. Jeder Fall nennt die verstellte Achse.
# fault_kind steuert die 3D-Schemaskizze.
CASES = [
    ("T_X_+05.000mm", "Verschobenes Teil", "shift_x",
     "5 mm in X verschoben (längs der Naht) — Abweichung nur an den Enden"),
    ("R_Y_+01.000deg", "Verkipptes Teil", "tilt_y",
     "1° um die Y-Achse gekippt (Spalt-Querachse)"),
    ("R_Z_+01.000deg", "Verdrehtes Teil", "yaw_z",
     "1° um die Z-Achse gedreht — der Spalt öffnet sich keilförmig"),
]
FOOTER = "Synthetische Daten"


def _plate_axes(ax):
    ax.set_aspect("equal")
    ax.set_xlabel("X — Naht-Längsrichtung (mm)", fontsize=9, color=INK)
    ax.set_ylabel("Y — quer (mm)", fontsize=9, color=INK)
    ax.tick_params(colors=INK, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


# ── 3D-Schemaskizze: welche Achse wurde verstellt ─────────────────────

# Schematische, überhöhte Bauteilmaße (einheitenlos). Naht läuft in X, die
# beiden Werkstücke sind in Y durch den Spalt getrennt, Dicke in Z.
_LX, _WY, _GAP, _TZ = 100.0, 30.0, 7.0, 8.0
C_WP_A = "#c3ccd4"   # Werkstück A (fest)
C_WP_B = "#9fb0bd"   # Werkstück B (verstellt)


def _box_polys(x0, x1, y0, y1, z0, z1, tf):
    """Sechs Quad-Flächen eines Quaders, Ecken durch tf transformiert."""
    pts = np.array([tf(np.array([x, y, z], float))
                    for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    i = lambda ix, iy, iz: ix * 4 + iy * 2 + iz
    faces = [
        [i(0, 0, 0), i(0, 1, 0), i(0, 1, 1), i(0, 0, 1)],
        [i(1, 0, 0), i(1, 1, 0), i(1, 1, 1), i(1, 0, 1)],
        [i(0, 0, 0), i(1, 0, 0), i(1, 0, 1), i(0, 0, 1)],
        [i(0, 1, 0), i(1, 1, 0), i(1, 1, 1), i(0, 1, 1)],
        [i(0, 0, 0), i(1, 0, 0), i(1, 1, 0), i(0, 1, 0)],
        [i(0, 0, 1), i(1, 0, 1), i(1, 1, 1), i(0, 1, 1)],
    ]
    return [pts[f] for f in faces]


def _rot_y_about(theta_deg, xp, zp):
    """Drehung um eine zu Y parallele Achse durch (xp, ·, zp)."""
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)

    def tf(p):
        x, y, z = p
        x, z = x - xp, z - zp
        return np.array([x * c + z * s + xp, y, -x * s + z * c + zp])
    return tf


def _rot_z_about(theta_deg, xp, yp):
    """Drehung um eine zu Z parallele Achse durch (xp, yp, ·) — Yaw."""
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)

    def tf(p):
        x, y, z = p
        x, y = x - xp, y - yp
        return np.array([x * c - y * s + xp, x * s + y * c + yp, z])
    return tf


def _draw_slab(ax, y0, y1, color, tf):
    polys = _box_polys(-_LX / 2, _LX / 2, y0, y1, 0, _TZ, tf)
    ax.add_collection3d(Poly3DCollection(
        polys, facecolor=color, edgecolor=INK2, linewidths=0.7, alpha=0.96))


def draw_fault_schematic(ax, kind: str):
    """Zwei Werkstücke; B (unten, y<0) trägt die überhöhte Fehlstellung."""
    ident = lambda p: p
    # Werkstück A (fest, oben, y>0)
    _draw_slab(ax, _GAP / 2, _GAP / 2 + _WY, C_WP_A, ident)

    # Werkstück B (unten, y<0) mit Fehlstellung, deutlich überhöht
    yb0, yb1 = -_GAP / 2 - _WY, -_GAP / 2
    dx_shift = 20.0
    d_xyz = np.array([16.0, -9.0, -6.0])   # überhöhte Kombi-Translation
    if kind == "clean":
        _draw_slab(ax, yb0, yb1, C_WP_B, ident)
    elif kind == "shift_x":
        _draw_slab(ax, yb0, yb1, C_WP_B,
                   lambda p: p + np.array([dx_shift, 0, 0]))
    elif kind == "shift_xyz":
        _draw_slab(ax, yb0, yb1, C_WP_B, lambda p: p + d_xyz)
    elif kind == "yaw_z_shift_x":
        # um Z gedreht (Spalt wird keilförmig) und zusätzlich in X verschoben
        yc = (yb0 + yb1) / 2
        rot = _rot_z_about(12, 0, yc)
        _draw_slab(ax, yb0, yb1, C_WP_B,
                   lambda p: rot(p) + np.array([16.0, 0, 0]))
    elif kind == "tilt_y":
        _draw_slab(ax, yb0, yb1, C_WP_B, _rot_y_about(15, 0, _TZ / 2))
    elif kind == "yaw_z":
        # Drehung um Z am rechten Nahtende: das linke Ende schwenkt in Y aus,
        # der Spalt öffnet sich keilförmig.
        _draw_slab(ax, yb0, yb1, C_WP_B, _rot_z_about(11, _LX / 2, yb1))

    # Koordinatensystem klar vor den Teilen (weit vorne links), damit kein
    # Werkstück es verdeckt — auch die +Y-Achse endet noch vor der Vorderkante.
    ox, oy, oz = -_LX / 2 - 12, yb0 - 26, 0
    triad = [((30, 0, 0), "X"), ((0, 16, 0), "Y"), ((0, 0, 20), "Z")]
    for (u, v, w), name in triad:
        ax.quiver(ox, oy, oz, u, v, w, color=INK, arrow_length_ratio=0.18,
                  linewidth=1.8, zorder=20)
        ax.text(ox + u * 1.18, oy + v * 1.25, oz + w * 1.16, name, fontsize=11,
                color=INK, weight="bold", ha="center", va="center", zorder=21)

    # Drehachse markieren
    if kind in ("tilt_y", "raise_tilt_y"):
        ax.plot([0, 0], [yb0 - 6, _GAP / 2 + _WY], [_TZ / 2, _TZ / 2],
                color=C_EXCESS, linewidth=1.8, linestyle=(0, (4, 2)), zorder=10)
        ax.text(6, _GAP / 2 + _WY + 2, _TZ / 2 + 6, "Drehachse Y",
                fontsize=8.5, color=C_EXCESS, weight="bold", ha="left")
    elif kind == "yaw_z":
        ax.plot([_LX / 2, _LX / 2], [yb1, yb1], [0, _TZ + 20],
                color=C_EXCESS, linewidth=1.8, linestyle=(0, (4, 2)), zorder=10)
        ax.text(_LX / 2 + 2, yb1, _TZ + 22, "Drehachse Z", fontsize=8.5,
                color=C_EXCESS, weight="bold", ha="left")
    elif kind == "shift_x":
        # roter Richtungspfeil der Translation, parallel zu +X über B,
        # rechts der Z-Achse beginnend, damit nichts überlappt
        yc = (yb0 + yb1) / 2
        ax.quiver(0, yc, _TZ + 9, 34, 0, 0, color=C_EXCESS,
                  arrow_length_ratio=0.26, linewidth=2.4, zorder=11)
        ax.text(0, yc, _TZ + 16, "verschoben in X", fontsize=8.5,
                color=C_EXCESS, weight="bold", ha="left", va="bottom")
    elif kind == "shift_xyz":
        # roter Pfeil oberhalb der Teile in Richtung der überhöhten
        # Kombi-Translation, damit er nicht in den Quadern verschwindet
        ax.quiver(-42, 8, _TZ + 18, 46, -17, -12, color=C_EXCESS,
                  arrow_length_ratio=0.2, linewidth=2.6, zorder=12)
        ax.text(-46, 8, _TZ + 25, "Translation X·Y·Z", fontsize=8.5,
                color=C_EXCESS, weight="bold", ha="left")
    elif kind == "yaw_z_shift_x":
        yc = (yb0 + yb1) / 2
        ax.plot([16, 16], [yc, yc], [0, _TZ + 24], color=C_EXCESS,
                linewidth=1.8, linestyle=(0, (4, 2)), zorder=10)
        ax.text(20, yc, _TZ + 27, "Drehachse Z", fontsize=8.5,
                color=C_EXCESS, weight="bold", ha="left")
        # roter X-Pfeil ohne Label — Richtung reicht, die Kastenunterschrift
        # nennt die Verschiebung; ein schwebendes 3D-Label würde clippen
        ax.quiver(-30, yc, _TZ + 7, 32, 0, 0, color=C_EXCESS,
                  arrow_length_ratio=0.26, linewidth=2.6, zorder=11)

    ax.set_box_aspect((_LX, 2 * _WY + _GAP, 40))
    ax.view_init(elev=20, azim=-62)
    ax.set_axis_off()
    ax.set_xlim(-_LX / 2 - 14, _LX / 2 + 24)
    ax.set_ylim(yb0 - 30, _GAP / 2 + _WY + 4)
    ax.set_zlim(-8, 32)


def _footer(fig):
    fig.text(0.5, 0.012, FOOTER, ha="center", fontsize=9, color=INK,
             style="italic")


# ── Bild 1: Abweichungs-Farbkarte ─────────────────────────────────────

def figure_deviation_map(cad, path: Path):
    fig = plt.figure(figsize=(13.5, 12.8), dpi=DPI)
    fig.subplots_adjust(left=0.03, right=0.86, top=0.85, bottom=0.05,
                        hspace=0.42, wspace=0.04)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-VMAX, vmax=VMAX)

    # Spaltenüberschriften
    fig.text(0.24, 0.925, "Was verstellt wurde", ha="center", fontsize=12,
             color=INK, weight="bold")
    fig.text(0.24, 0.910, "Schema, Fehlstellung überhöht", ha="center",
             fontsize=9, color=INK)
    fig.text(0.64, 0.925, "Was die Messung sieht", ha="center", fontsize=12,
             color=INK, weight="bold")
    fig.text(0.64, 0.910, "Draufsicht auf die XY-Ebene", ha="center",
             fontsize=9, color=INK)

    mappable = None
    for r, (case, title, fault, sub) in enumerate(CASES):
        ax3 = fig.add_subplot(3, 2, 2 * r + 1, projection="3d")
        draw_fault_schematic(ax3, fault)

        axm = fig.add_subplot(3, 2, 2 * r + 2)
        field = signed_distance_field(OUTPUTS / case, cad)
        sel = subsample(len(field.points), 45000, seed=2)
        p, d = field.points[sel], field.signed[sel]
        order = np.argsort(np.abs(d))
        mappable = axm.scatter(p[order, 0], p[order, 1], c=d[order],
                               cmap=DEV_CMAP, norm=norm, s=2, marker="s",
                               linewidths=0, rasterized=True)
        rate = field.in_tolerance_rate * 100
        axm.set_title(f"{title} — {rate:.0f} % in Toleranz", fontsize=12,
                      color=INK, weight="bold", pad=14)
        axm.text(0.5, 1.02, sub, transform=axm.transAxes, ha="center",
                 va="bottom", fontsize=9.5, color=INK)
        _plate_axes(axm)

    fig.suptitle("Abweichung zum CAD-Ideal — Scan eingefärbt nach Abstand",
                 fontsize=15, weight="bold", color=INK, y=0.975)

    cax = fig.add_axes([0.895, 0.30, 0.015, 0.40])
    cbar = fig.colorbar(mappable, cax=cax, extend="both")
    cbar.ax.tick_params(colors=INK, labelsize=8)
    cbar.ax.axhspan(-TOLERANCE_MM, TOLERANCE_MM, facecolor="none",
                    edgecolor=INK, linewidth=1.0)
    for y, txt, va in ((VMAX, "Material\nsteht über", "top"),
                       (0.0, "in Toleranz\n(±0.25 mm)", "center"),
                       (-VMAX, "Material\nfehlt", "bottom")):
        cbar.ax.text(2.6, y, txt, transform=cbar.ax.get_yaxis_transform(),
                     ha="left", va=va, fontsize=8.5, color=INK)

    _footer(fig)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── Bild 2: C_TR_08-Dreiklang ─────────────────────────────────────────

def _load_report(case: str) -> dict:
    with open(OUTPUTS / case / "subtraction_report.json", encoding="utf-8") as fh:
        return json.load(fh)


def figure_three_levels(cad, path: Path):
    # Drehung um Z + Verschiebung in X: beide Fehler leben in der Draufsicht
    # (der Spalt wird keilförmig, die Enden wandern), kein Höhenanteil, der
    # erst in 3D sichtbar wäre. Eigens erzeugter Fall, Rotation um B's
    # Schwerpunkt, damit die sechs Freiheitsgrade sauber Rz + Tx zeigen.
    case = "C_ZX_01"
    if not (OUTPUTS / case / "subtraction_report.json").exists():
        raise SystemExit(
            f"Fall {case} fehlt in data/outputs/. Erst erzeugen:\n"
            "  uv run python scripts/generate_rztx_demo.py\n"
            "  uv run python scripts/run_batch_subtraction.py "
            "--scan-dir data/raw/_rztx_demo --source-type synthetic")
    rep = _load_report(case)["deviation"]
    field = signed_distance_field(OUTPUTS / case, cad)
    rate = field.in_tolerance_rate * 100
    cr = rep["component_registration"]
    tr, rot = cr["translation_mm"], cr["rotation_deg"]

    fig = plt.figure(figsize=(18, 6.9), dpi=DPI)
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_axis_off()
    bg.set_xlim(0, 1); bg.set_ylim(0, 1)

    # Vier gleich hohe Kästen: der Fehler und die drei Auswertungsebenen.
    cards = {0: (0.020, 0.10, 0.225, 0.72),   # Schema
             1: (0.262, 0.10, 0.170, 0.72),   # Global
             2: (0.449, 0.10, 0.258, 0.72),   # Voxel
             3: (0.724, 0.10, 0.256, 0.72)}   # 6 DOF
    for x, y, w, h in cards.values():
        bg.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.016",
            facecolor="#f6f7f9", edgecolor="#ccd3da", linewidth=1.4))
    cx = lambda i: cards[i][0] + cards[i][2] / 2
    header_y = cards[0][1] + cards[0][3] - 0.055
    for i, htext in ((0, "Der Fehler"), (1, "① Global"),
                     (2, "② Voxel"), (3, "③ Sechs Freiheitsgrade")):
        bg.text(cx(i), header_y, htext, ha="center", fontsize=15.5,
                weight="bold", color=INK)

    # ── Kasten 0: Schemaskizze (innerhalb der Kastengrenzen) ─────────
    x, y, w, h = cards[0]
    ax3 = fig.add_axes([x + 0.004, 0.30, w - 0.008, 0.42], projection="3d")
    draw_fault_schematic(ax3, "yaw_z_shift_x")
    bg.text(cx(0), 0.25, "Werkstück um Z gedreht\nund in X verschoben.",
            ha="center", va="top", fontsize=12.5, weight="bold", color=INK)

    # ── Kasten 1: globaler Kennwert ──────────────────────────────────
    bg.text(cx(1), 0.55, f"{rate:.0f} %", ha="center", va="center",
            fontsize=54, weight="bold", color=C_BAD)
    bg.text(cx(1), 0.445, "in Toleranz", ha="center", fontsize=15,
            weight="bold", color=INK)
    x, y, w, h = cards[1]
    bx, bw, by = x + 0.022, w - 0.044, 0.37
    bg.add_patch(plt.Rectangle((bx, by), bw, 0.026, color="#d5dae0"))
    bg.add_patch(plt.Rectangle((bx, by), bw * rate / 100, 0.026, color=C_BAD))
    bg.text(cx(1), 0.30, "Anteil der Fläche innerhalb\n±0.25 mm — ein Skalar\n"
            "ohne räumliche Auflösung.", ha="center", va="top", fontsize=12.5,
            color=INK)

    # ── Kasten 2: Voxelkarte ─────────────────────────────────────────
    x, y, w, h = cards[2]
    axv = fig.add_axes([x + 0.008, 0.335, w - 0.016, 0.40])
    vox = rep["voxel_deviation"]
    centers = np.array(vox["centers"])
    mean_signed = np.array(vox["mean_signed"], dtype=float)
    # enge Skala: die Keil-Abweichung sitzt konzentriert an der Naht, eine
    # weite Skala würde sie ausbleichen.
    vmax = 0.4
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    axv.scatter(centers[:, 0], centers[:, 1], c=mean_signed, cmap=DEV_CMAP,
                norm=norm, s=14, marker="s", linewidths=0)
    axv.set_aspect("equal"); axv.set_axis_off()
    bg.text(cx(2), 0.285, "Mittlerer Abstand je 5-mm-Würfel —\n"
            "lokalisiert die Abweichung.", ha="center", va="top",
            fontsize=12.5, color=INK)
    bg.text(cx(2), 0.17, "rot: Material steht über   ·   blau: fehlt",
            ha="center", fontsize=12, weight="bold", color=INK)

    # ── Kasten 3: sechs Freiheitsgrade ───────────────────────────────
    x, y, w, h = cards[3]
    bg.text(cx(3), 0.66, "Relativlage der Werkstücke, aus der Registrierung",
            ha="center", fontsize=11.5, color=INK)
    col_t, col_r = x + 0.03, x + 0.145
    bg.text(col_t, 0.585, "Translation", fontsize=12, color=INK, weight="bold")
    bg.text(col_r, 0.585, "Rotation", fontsize=12, color=INK, weight="bold")
    dof = [("Tx", tr["x"], "mm"), ("Ty", tr["y"], "mm"), ("Tz", tr["z"], "mm"),
           ("Rx", rot["x"], "°"), ("Ry", rot["y"], "°"), ("Rz", rot["z"], "°")]
    for k, (name, val, unit) in enumerate(dof):
        col = col_t if k < 3 else col_r
        yy = 0.50 - (k % 3) * 0.115
        disp = 0.0 if abs(val) < 0.005 else val    # kein "-0.00"
        # alles schwarz; nur die echten Fehler (>0.1) fett, der Rest normal
        wt = "bold" if abs(val) > 0.1 else "normal"
        bg.text(col, yy, name, fontsize=13, color=INK, weight="bold")
        bg.text(col + 0.028, yy, f"{disp:+.2f} {unit}", fontsize=15,
                color=INK, weight=wt)

    fig.suptitle("Drei Auswertungsebenen am selben Bauteil — Drehung um Z "
                 "und Verschiebung in X", fontsize=16, weight="bold",
                 color=INK, y=0.95)
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
    ax.set_ylabel("Anteil der Naht in Toleranz (%)", fontsize=10, color=INK)
    ax.set_title("Wie viel liegt in Toleranz?", fontsize=12, color=INK,
                 weight="bold")

    ax = axes[1]
    bars = ax.bar(labels, rms, color=colors, width=0.6, zorder=3)
    for b, v in zip(bars, rms):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.4f} mm",
                ha="center", fontsize=13, color=INK, weight="bold")
    ax.set_ylim(0, 0.47)
    ax.set_ylabel("Registrierungs-Einfluss RMS (mm)", fontsize=10, color=INK)
    ax.set_title("Wie stark verfälscht die Ausrichtung?", fontsize=12,
                 color=INK, weight="bold")
    ax.annotate("240× kleiner", xy=(1, rms[1]), xytext=(1, 0.22),
                ha="center", fontsize=11, color=C_GOOD, weight="bold",
                arrowprops=dict(arrowstyle="->", color=C_GOOD, lw=1.6))

    for ax in axes:
        ax.tick_params(colors=INK, labelsize=10)
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
