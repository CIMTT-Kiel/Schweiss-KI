#!/usr/bin/env python3
"""
Erzeugt die Abbildungen für docs/methodik_abweichungsanalyse.md.

Die Bilder sind schematisch, nicht datengetrieben – sie erklären die
Geometrie und das Verankerungsprinzip. Messwerte gehören in die
Fehleranalyse-Doku, nicht hierher.

Aufruf:  uv run python scripts/generate_methodik_figures.py
Ausgabe: docs/figures/*.png (dpi=200, kein SVG – rendert in PowerPoint fehlerhaft)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from schweiss_ki.core.console import force_utf8_output

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
DPI = 200

# Palette konsistent zu src/schweiss_ki/subtraction/plots.py – keine neuen
# kategorialen Farben. Flanken tragen ihre Label-Farbe, Bezugsgrößen
# (Referenzebene, Hilfslinien) neutrale Ink-Tokens.
C_FLANK_A = "#1976D2"
C_FLANK_B = "#FF9800"
C_REF = "#37474F"
C_GUIDE = "#B0BEC5"
C_BODY = "#CFD8DC"
C_BAD = "#E53935"

# Nahtgeometrie (schematisch) – Wurzelspalt und Blechdicke wie beim
# Referenzbauteil Baugruppe_Beispielteil_V-Naht_1.5mm_Spalt.
# Wichtig: die Flanken laufen NICHT spitz zu, sie enden am Wurzelspalt.
ROOT_GAP = 1.5
THICK = 5.0
ALPHA = 45.0
HALF_W = 18.0
SEAM_LEN = 60.0
TAN_A = np.tan(np.deg2rad(ALPHA))
TOP_EDGE = ROOT_GAP / 2 + THICK * TAN_A     # Fasenkante an der Oberseite


def _style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ── Bild 1: Achsenkonvention ──────────────────────────────────────────

def figure_axes_convention(path: Path):
    """Zwei Projektionen statt einer Isometrie.

    Eine 3D-Ansicht ist fuer eine Achsenkonvention umstaendlicher als zwei
    saubere Schnitte: Draufsicht zeigt, dass die Naht entlang X laeuft,
    Querschnitt zeigt Spaltrichtung und Tiefe.
    """
    fig, (ax_top, ax_cs) = plt.subplots(
        1, 2, figsize=(11, 4.4), dpi=DPI,
        gridspec_kw={"width_ratios": [1.5, 1.0]})

    # ── Draufsicht XY ────────────────────────────────────────────────
    for sign, c in ((-1, C_FLANK_A), (+1, C_FLANK_B)):
        ax_top.fill_between([0, SEAM_LEN], sign * TOP_EDGE, sign * HALF_W,
                            color=C_BODY, alpha=0.6, linewidth=0)
        ax_top.fill_between([0, SEAM_LEN], sign * ROOT_GAP / 2, sign * TOP_EDGE,
                            color=c, alpha=0.55, linewidth=0)
    ax_top.fill_between([0, SEAM_LEN], -ROOT_GAP / 2, ROOT_GAP / 2,
                        color="white", linewidth=0)
    ax_top.annotate("", xy=(SEAM_LEN * 0.92, 0), xytext=(SEAM_LEN * 0.08, 0),
                    arrowprops=dict(arrowstyle="->", color=C_REF, lw=2.0))
    ax_top.text(SEAM_LEN * 0.5, 1.4, "Naht verläuft entlang X",
                color=C_REF, fontsize=10, ha="center", weight="bold")
    ax_top.text(SEAM_LEN * 0.5, -TOP_EDGE - 3.5, "Werkstück mit Flanke A",
                color=C_FLANK_A, fontsize=9.5, ha="center", weight="bold")
    ax_top.text(SEAM_LEN * 0.5, TOP_EDGE + 2.2, "Werkstück mit Flanke B",
                color=C_FLANK_B, fontsize=9.5, ha="center", weight="bold")
    ax_top.set_xlim(0, SEAM_LEN)
    ax_top.set_ylim(-HALF_W, HALF_W)
    # Bewusst NICHT aspect="equal": die Draufsicht ist schematisch und zeigt
    # eine Richtung, keine wahren Proportionen. Mit gleichem Achsenmassstab
    # waeren die beiden Panels stark unterschiedlich hoch.
    _style(ax_top, "X — Naht-Längsrichtung (mm)", "Y — Spalt-Querrichtung (mm)")
    ax_top.set_title("Draufsicht (XY)", fontsize=12)

    # ── Querschnitt YZ ───────────────────────────────────────────────
    x_lim = 10.5
    for sign, c in ((-1, C_FLANK_A), (+1, C_FLANK_B)):
        y_top, y_out, y_root = sign * TOP_EDGE, sign * x_lim, sign * ROOT_GAP / 2
        ax_cs.fill([y_root, y_top, y_out, y_out, y_root],
                   [-THICK, 0, 0, -THICK - 0.8, -THICK - 0.8],
                   color=C_BODY, alpha=0.6, linewidth=0)
        ax_cs.plot([y_root, y_top], [-THICK, 0], color=c, linewidth=3.0)
    ax_cs.annotate("", xy=(-ROOT_GAP / 2, -THICK - 0.45),
                   xytext=(ROOT_GAP / 2, -THICK - 0.45),
                   arrowprops=dict(arrowstyle="<->", color=C_REF, lw=1.6))
    ax_cs.text(0, -THICK - 1.5, "Wurzelspalt", color=C_REF, fontsize=9,
               ha="center", weight="bold")
    ax_cs.text(-TOP_EDGE - 0.3, -THICK * 0.45, "Flanke A", color=C_FLANK_A,
               fontsize=10, ha="right", weight="bold")
    ax_cs.text(TOP_EDGE + 0.3, -THICK * 0.45, "Flanke B", color=C_FLANK_B,
               fontsize=10, ha="left", weight="bold")
    ax_cs.set_xlim(-x_lim, x_lim)
    ax_cs.set_ylim(-THICK - 2.0, 1.0)
    ax_cs.set_aspect("equal")
    _style(ax_cs, "Y — Spalt-Querrichtung (mm)", "Z — Tiefe (mm)")
    ax_cs.set_title("Querschnitt (YZ)", fontsize=12)

    fig.suptitle("Achsenkonvention", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── Bild 2: Querschnitt mit Verankerung ───────────────────────────────

def _flank_xy(sign, d_lo, d_hi):
    """Flankenlinie als Funktion der Tiefe d unter der Deckfläche."""
    d = np.array([d_lo, d_hi])
    y = sign * (ROOT_GAP / 2 + (THICK - d) * TAN_A)
    return y, -d


def figure_cross_section(path: Path):
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=DPI)
    d_min, d_root = 0.5, THICK - 0.25
    x_lim = 10.5

    # Werkstückkörper, flach geschlossen (nicht sloped)
    for sign in (-1, +1):
        y_top, y_out = sign * TOP_EDGE, sign * x_lim
        y_root = sign * ROOT_GAP / 2
        ax.fill([y_root, y_top, y_out, y_out, y_root],
                [-THICK, 0, 0, -THICK - 0.8, -THICK - 0.8],
                color=C_BODY, alpha=0.6, zorder=1, linewidth=0)

    # Referenzebene (der Anker)
    ax.axhline(0, color=C_REF, linewidth=2.2, zorder=6)
    ax.text(0, 0.55, "Referenzebene — Deckfläche Werkstück A   (d = 0)",
            color=C_REF, fontsize=9.5, weight="bold", ha="center")

    # Flankenprofile über ihrem Fitband
    for sign, c, name in ((-1, C_FLANK_A, "Flankenprofil A"),
                          (+1, C_FLANK_B, "Flankenprofil B")):
        y, z = _flank_xy(sign, d_min, d_root)
        ax.plot(y, z, color=c, linewidth=2.6, zorder=8, label=name)
        y_full, z_full = _flank_xy(sign, 0.0, THICK)
        ax.plot(y_full, z_full, color=c, linewidth=1.0, alpha=0.35,
                linestyle="--", zorder=4)

    # Fitband
    for d, txt in ((d_min, "Fitband-Obergrenze"), (d_root, "Wurzeltiefe  $d_{root}$")):
        ax.axhline(-d, color=C_GUIDE, linestyle=":", linewidth=1.3, zorder=3)
        ax.text(x_lim - 0.3, -d + 0.18, txt, color=C_REF, fontsize=8.5, ha="right")

    # Tiefenachse
    ax.annotate("", xy=(-x_lim + 1.5, -d_root), xytext=(-x_lim + 1.5, 0),
                arrowprops=dict(arrowstyle="<->", color=C_REF, lw=1.3))
    ax.text(-x_lim + 2.0, -d_root / 2, "Tiefe $d$\nunter der\nReferenzebene",
            color=C_REF, fontsize=8.5, va="center")

    # Gemessener Spalt an der Wurzel
    y_a = -(ROOT_GAP / 2 + (THICK - d_root) * TAN_A)
    y_b = +(ROOT_GAP / 2 + (THICK - d_root) * TAN_A)
    ax.annotate("", xy=(y_b, -d_root), xytext=(y_a, -d_root),
                arrowprops=dict(arrowstyle="<->", color=C_REF, lw=2.0))
    ax.annotate("Spalt = Differenz der\nbeiden Flankenprofile",
                xy=(0.6, -d_root), xytext=(5.6, -THICK - 0.35),
                color=C_REF, fontsize=9, ha="center", weight="bold",
                arrowprops=dict(arrowstyle="->", color=C_REF, lw=1.0))

    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(-THICK - 1.1, 1.15)
    ax.set_aspect("equal")
    _style(ax, "Y — Spalt-Querrichtung (mm)", "Z (mm)")
    ax.set_title("Deckflächenverankerte Spaltmessung im Querschnitt",
                 fontsize=13, pad=14)
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.80),
              fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── Bild 3: Verstärkungseffekt ────────────────────────────────────────

def figure_amplification(path: Path):
    """Warum eine feste Auswertungshöhe den Spalt verfälscht.

    Beide Panels zeigen DASSELBE Bauteil in derselben (um dz zu hoch
    registrierten) Lage. Unterschiedlich ist allein, woran die
    Auswertungshöhe hängt – und damit, in welcher TIEFE im Bauteil
    geschnitten wird. Genau diese Tiefe ist deshalb in beiden Panels
    bemaßt: sie ist der Mechanismus, nicht die Lage der Linie.
    """
    dz = 0.6
    # Ausgewertet wird an der WURZEL – die Groesse, die das Verfahren liefert.
    # Das Bauteil ist hier um dz zu TIEF registriert; dadurch bleiben beide
    # Schnitte im Material. (Bei zu hoher Lage laege der feste Schnitt
    # unterhalb der Wurzel, also ausserhalb des Bauteils.)
    d_nom = THICK
    z_deck = -dz   # gemessene Deckfläche, um dz zu tief
    x_depth = 7.2

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), dpi=DPI, sharey=True)

    def draw(ax, anchored):
        # Soll-Höhe der Deckfläche (wo das Bauteil liegen sollte)
        ax.axhline(0, color=C_GUIDE, linestyle=":", linewidth=1.3, zorder=2)
        # Tatsächlich gemessene Deckfläche
        ax.axhline(z_deck, color=C_REF, linewidth=2.2, zorder=6)
        for sign, c in ((-1, C_FLANK_A), (+1, C_FLANK_B)):
            y, z = _flank_xy(sign, 0.0, THICK)
            ax.plot(y, z + z_deck, color=c, linewidth=2.8, zorder=8)

        if anchored:
            z_anchor = z_deck                # Bezug: Deckfläche des Bauteils
            z_eval = z_deck - d_nom
            farbe, titel = C_REF, "verankert an der Deckfläche"
            anchor_txt = "Bezug:\nDeckfläche"
            mess_txt = "= Wurzelspalt"
            note = ("Bezug wandert mit dem Bauteil\n"
                    "→ Schnitt trifft die Wurzel, Spalt korrekt")
        else:
            z_anchor = 0.0                   # Bezug: Koordinatensystem
            z_eval = -d_nom
            farbe, titel = C_BAD, "feste Auswertungshöhe"
            anchor_txt = "Bezug:\nz = 0"
            mess_txt = "nicht die Wurzel"
            note = (f"Bezug bleibt bei z = 0, das Bauteil sitzt tiefer\n"
                    f"→ Schnitt nur {THICK - dz:.1f} mm tief statt {THICK:.1f} — "
                    f"Spalt 2·dz = {2 * dz:.1f} mm zu breit")

        d_actual = z_deck - z_eval
        half = ROOT_GAP / 2 + (THICK - d_actual) * TAN_A

        # Kernaussage: gleiche Bemaßung, unterschiedlicher Startpunkt.
        ax.plot([x_depth - 0.6, x_depth + 0.6], [z_anchor, z_anchor],
                color=farbe, linewidth=2.0, zorder=9)
        ax.annotate("", xy=(x_depth, z_eval), xytext=(x_depth, z_anchor),
                    arrowprops=dict(arrowstyle="<->", color=farbe, lw=1.8))
        ax.text(x_depth + 0.5, (z_anchor + z_eval) / 2, f"{d_nom:.1f} mm",
                color=farbe, fontsize=10, va="center", weight="bold")
        ax.text(x_depth + 0.5, z_anchor + 0.15, anchor_txt,
                color=farbe, fontsize=8.5, va="bottom")

        # Auswertungshöhe und gemessene Spaltbreite
        ax.axhline(z_eval, color=farbe, linestyle="--", linewidth=1.8, zorder=7)
        ax.annotate("", xy=(half, z_eval), xytext=(-half, z_eval),
                    arrowprops=dict(arrowstyle="<->", color=farbe, lw=2.2))
        ax.text(half + 0.55, z_eval, f"{2 * half:.1f} mm\n{mess_txt}", color=farbe,
                fontsize=10.5, ha="left", va="center", weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.9))

        ax.text(0.5, 0.03, note, transform=ax.transAxes, fontsize=8.5,
                ha="center", color=farbe,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor=farbe, alpha=0.95))
        ax.set_title(titel, fontsize=12, color=farbe, weight="bold", pad=8)
        ax.set_xlim(-9.8, 11.0)
        ax.set_ylim(-THICK - 3.4, 1.5)
        ax.set_aspect("equal")
        _style(ax, "Y — Spalt-Querrichtung (mm)", "")

    draw(axes[0], anchored=False)
    draw(axes[1], anchored=True)
    axes[0].set_ylabel("Z (mm)")
    # Nur im linken Panel beschriften – rechts ist die Lage dieselbe.
    axes[0].text(-9.6, 0.45, "Soll-Höhe der Deckfläche",
                 color="#8A9AA5", fontsize=8.5)
    # Schmal gesetzt: breiter Text würde in Flanke A hineinlaufen.
    axes[0].text(-9.6, -1.5, f"gemessene\nDeckfläche\n({dz} mm zu tief)",
                 color=C_REF, fontsize=8.5, weight="bold", va="top",
                 linespacing=1.4)
    fig.suptitle(
        "Beide Seiten: dasselbe Bauteil, um dz zu tief registriert — "
        "nur der Bezug der Auswertungshöhe unterscheidet sich",
        fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    force_utf8_output()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (
        ("01_achsenkonvention.png", figure_axes_convention),
        ("02_querschnitt_verankerung.png", figure_cross_section),
        ("03_verstaerkungseffekt.png", figure_amplification),
    ):
        fn(OUT / name)
        print(f"  {OUT / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
