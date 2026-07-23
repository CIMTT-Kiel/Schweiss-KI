#!/usr/bin/env python3
"""Wertet den synthetischen Batch-Lauf aus und erzeugt die Validierungsplots.

Drei Auswertungen, entsprechend den drei Ebenen der Abweichungsanalyse:

  1. Restfehler über alle Fälle, nach Fehlerklasse getrennt, gegen ±0.25 mm
  2. Merkmale über die Serie, gegen die bekannte Ground Truth geprüft
  3. Alle drei Ebenen nebeneinander für einen sauberen und einen schweren Fall

Aufruf:  uv run python scripts/analyze_synthetic_validation.py
Ausgabe: docs/figures/validierung/*.png  +  Zusammenfassung auf stdout
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from schweiss_ki.core.console import force_utf8_output
from schweiss_ki.analysis.synthetic_validation import (
    CLASS_COLORS, CLASS_ORDER, NOMINAL_FLANK_ANGLE_DEG, TOLERANCE_MM,
    load_results,
)

OUTPUTS = ROOT / "data" / "outputs"
META = ROOT / "data" / "raw" / "synthetic_scans" / "synthetic_metadata.csv"
FIGDIR = ROOT / "docs" / "figures" / "validierung"
DPI = 200

C_REF, C_GUIDE, C_BAD = "#37474F", "#B0BEC5", "#E53935"
CASE_CLEAN, CASE_HARD = "T_X_+00.100mm", "C_TR_08"


def _style(ax, xlabel="", ylabel="", title=""):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _jitter(n, width=0.28, seed=0):
    """Reproduzierbarer Versatz, damit sich Punkte gleicher Klasse nicht decken."""
    return np.random.default_rng(seed).uniform(-width, width, n)


def _tolerance_band(ax, tol=TOLERANCE_MM, label=True):
    ax.axhspan(-tol, tol, color=C_GUIDE, alpha=0.30, zorder=0,
               label="±0.25 mm Toleranz" if label else None)
    ax.axhline(0, color=C_REF, linewidth=1.0, zorder=1)


# ── Auswertung 1: Restfehler nach Fehlerklasse ────────────────────────

def figure_error_distribution(df: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), dpi=DPI)

    # (a) Alle 61 Faelle. Offene Marker: dort beschreibt das Sollmodell die
    # Geometrie nicht vollstaendig (ry/rz erzeugen einen laengs veraenderlichen
    # Spalt), der Wert ist also kein Messfehler.
    ax = axes[0]
    _tolerance_band(ax)
    for i, cls in enumerate(CLASS_ORDER):
        sub = df[df.error_class == cls]
        full, part = sub[sub.gap_model_complete], sub[~sub.gap_model_complete]
        ax.scatter(i + _jitter(len(full), seed=i), full.gap_residual_mm,
                   color=CLASS_COLORS[cls], s=34, alpha=0.85, linewidths=0,
                   zorder=3, label=f"{cls} (n={len(sub)})")
        ax.scatter(i + _jitter(len(part), seed=i + 40), part.gap_residual_mm,
                   facecolors="none", edgecolors=CLASS_COLORS[cls], s=40,
                   linewidths=1.3, zorder=3)
    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER)
    _style(ax, ylabel="Spaltbreite: gemessen − erwartet (mm)",
           title="(a) alle 61 — offen: Sollmodell unvollständig (ry/rz)")
    ax.legend(fontsize=8, loc="lower left")

    # (b) Nur die Faelle, fuer die das Sollmodell vollstaendig ist. Erst hier
    # ist der Rest tatsaechlich ein Messfehler.
    ax = axes[1]
    sub_all = df[df.gap_model_complete]
    for i, cls in enumerate(CLASS_ORDER):
        sub = sub_all[sub_all.error_class == cls]
        ax.scatter(i + _jitter(len(sub), seed=i), sub.gap_residual_mm,
                   color=CLASS_COLORS[cls], s=34, alpha=0.85, linewidths=0,
                   zorder=3, label=f"{cls} (n={len(sub)})")
    med = sub_all.gap_residual_mm.median()
    ax.axhline(med, color=C_BAD, linestyle="--", linewidth=1.4, zorder=4,
               label=f"Median {med:+.4f} mm")
    ax.axhline(0, color=C_REF, linewidth=1.0, zorder=1)
    ax.set_ylim(-0.07, 0.07)
    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER)
    _style(ax, ylabel="Restfehler (mm)",
           title=f"(b) nur vollständiges Sollmodell (n={len(sub_all)}), ±0.07 mm")
    ax.legend(fontsize=8, loc="upper right")

    # (c) Globale Distanz – hier wirkt die Registrierung, nicht die Verankerung
    ax = axes[2]
    for i, cls in enumerate(CLASS_ORDER):
        sub = df[df.error_class == cls]
        ax.scatter(i + _jitter(len(sub), seed=i), sub.global_mean_abs,
                   color=CLASS_COLORS[cls], s=34, alpha=0.8, linewidths=0, zorder=3)
    ax.axhline(TOLERANCE_MM, color=C_BAD, linestyle="--", linewidth=1.4,
               label="0.25 mm", zorder=4)
    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_yscale("log")
    _style(ax, ylabel="mittlerer |Abstand| zum CAD (mm)",
           title="(c) Globale Distanz (log)")
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Restfehler über 61 synthetische Fälle, nach Fehlerklasse",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── Auswertung 2: Merkmale gegen Ground Truth ─────────────────────────

def figure_features(df: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), dpi=DPI)

    # (a) Flankenwinkel je Seite
    ax = axes[0, 0]
    for i, cls in enumerate(CLASS_ORDER):
        sub = df[df.error_class == cls]
        for k, (col, mark) in enumerate((("flank_a_angle_deg", "o"),
                                         ("flank_b_angle_deg", "^"))):
            ax.scatter(i - 0.18 + 0.36 * k + _jitter(len(sub), 0.12, seed=i + k),
                       sub[col], color=CLASS_COLORS[cls], s=30, alpha=0.8,
                       marker=mark, linewidths=0, zorder=3)
    ax.axhline(NOMINAL_FLANK_ANGLE_DEG, color=C_REF, linestyle="--", linewidth=1.4,
               label="Soll 45°", zorder=4)
    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER)
    _style(ax, ylabel="Flankenwinkel (°)",
           title="(a) Flankenwinkel — ○ Flanke A, △ Flanke B")
    ax.legend(fontsize=8)

    # (b) Die Winkelsumme ist die eigentlich stabile Groesse
    ax = axes[0, 1]
    lim = [df.rx_deg.min() - 0.3, df.rx_deg.max() + 0.3]
    ax.plot(lim, lim, color=C_REF, linestyle="--", linewidth=1.4, label="Identität")
    for cls in CLASS_ORDER:
        sub = df[df.error_class == cls]
        ax.scatter(sub.rx_deg, sub.angle_sum_err, color=CLASS_COLORS[cls],
                   s=34, alpha=0.85, linewidths=0, zorder=3, label=cls)
    _style(ax, xlabel="aufgeprägte Querverkippung rx (°)",
           ylabel="(α_A − 45°) + (α_B − 45°)",
           title="(b) Winkelsumme folgt rx exakt")
    ax.legend(fontsize=8)

    # (c) Kantenversatz gegen aufgepraegtes tz. Nur rotationsfreie Faelle:
    # ein verkipptes Werkstueck hat konstruktionsbedingt ebenfalls einen
    # mittleren Hoehenversatz, dort ist tz allein nicht der Sollwert.
    ax = axes[0, 2]
    pure_t = df[(df.rx_deg == 0) & (df.ry_deg == 0) & (df.rz_deg == 0)]
    lim = [pure_t.tz_mm.min() - 0.2, pure_t.tz_mm.max() + 0.2]
    ax.plot(lim, lim, color=C_REF, linestyle="--", linewidth=1.4, label="Identität")
    ax.scatter(pure_t.tz_mm, pure_t.edge_offset_mm, color=CLASS_COLORS["Translation"],
               s=36, alpha=0.85, linewidths=0, zorder=3,
               label=f"ohne Rotation (n={len(pure_t)})")
    bias = (pure_t.edge_offset_mm - pure_t.tz_mm).mean()
    _style(ax, xlabel="aufgeprägtes tz (mm)", ylabel="gemessener Kantenversatz (mm)",
           title=f"(c) Kantenversatz — Bias {bias:+.4f} mm")
    ax.legend(fontsize=8)

    # (d) Relative Verkippung gegen die aufgepraegte Rotation
    ax = axes[1, 0]
    ax.plot(lim, lim, color=C_REF, linestyle="--", linewidth=1.4)
    ax.scatter(df.rx_deg, -df.tilt_across_gap_deg, color=CLASS_COLORS["Translation"],
               s=32, alpha=0.8, linewidths=0, zorder=3, label="quer zum Spalt vs. rx")
    ax.scatter(df.ry_deg, df.tilt_along_seam_deg, color=CLASS_COLORS["Rotation"],
               s=32, alpha=0.8, marker="^", linewidths=0, zorder=3,
               label="längs der Naht vs. ry")
    lo = min(df.rx_deg.min(), df.ry_deg.min()) - 0.3
    hi = max(df.rx_deg.max(), df.ry_deg.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], color=C_REF, linestyle="--", linewidth=1.4)
    _style(ax, xlabel="aufgeprägter Winkel (°)", ylabel="gemessene Verkippung (°)",
           title="(d) Relative Verkippung")
    ax.legend(fontsize=8)

    # (g) Keilsteigung: bei Verkippung um die Hochachse oeffnet sich der Spalt
    # laengs der Naht. Die Steigung ist dann der pruefbare Kennwert.
    ax = axes[0, 3]
    rz = df[df.rz_deg != 0]
    lim = [min(df.expected_wedge_slope.min(), -0.001) * 1.15,
           max(df.expected_wedge_slope.max(), 0.001) * 1.15]
    ax.plot(lim, lim, color=C_REF, linestyle="--", linewidth=1.4, label="Identität")
    for cls in CLASS_ORDER:
        sub = rz[rz.error_class == cls]
        ax.scatter(sub.expected_wedge_slope, sub.gap_wedge_slope,
                   color=CLASS_COLORS[cls], s=36, alpha=0.85, linewidths=0,
                   zorder=3, label=cls)
    _style(ax, xlabel="erwartet: −tan(rz)", ylabel="gemessene Keilsteigung (mm/mm)",
           title=f"(g) Spaltkeil bei rz ≠ 0 (n={len(rz)})")
    ax.legend(fontsize=8)

    # (h) Bin-Abdeckung — belegt, wieviel Nahtlaenge auswertbar war
    ax = axes[1, 3]
    for i, cls in enumerate(CLASS_ORDER):
        sub = df[df.error_class == cls]
        ax.scatter(i + _jitter(len(sub), seed=i), 100 * sub.bin_coverage,
                   color=CLASS_COLORS[cls], s=34, alpha=0.8, linewidths=0, zorder=3)
    ax.set_ylim(60, 104)
    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER)
    _style(ax, ylabel="ausgewertete Bins (%)", title="(h) Abdeckung entlang der Naht")

    # (e) Flankenasymmetrie je Klasse
    ax = axes[1, 1]
    for i, cls in enumerate(CLASS_ORDER):
        sub = df[df.error_class == cls]
        ax.scatter(i + _jitter(len(sub), seed=i), sub.flank_asymmetry_deg,
                   color=CLASS_COLORS[cls], s=34, alpha=0.8, linewidths=0, zorder=3)
    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER)
    _style(ax, ylabel="|α_A − α_B| (°)", title="(e) Flankenasymmetrie")

    # (f) Fit-Guete
    ax = axes[1, 2]
    for i, cls in enumerate(CLASS_ORDER):
        sub = df[df.error_class == cls]
        ax.scatter(i + _jitter(len(sub), seed=i), 1.0 - sub.fit_quality_min_r2,
                   color=CLASS_COLORS[cls], s=34, alpha=0.8, linewidths=0, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER)
    _style(ax, ylabel="1 − R² (schlechtere Flanke, log)", title="(f) Fit-Güte")

    fig.suptitle("Merkmale über die Serie, gegen die bekannte Ground Truth",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── Auswertung 3: drei Ebenen für zwei Fälle ──────────────────────────

def _load_voxels(case: str) -> dict:
    with open(OUTPUTS / case / "subtraction_report.json", encoding="utf-8") as fh:
        return json.load(fh)["deviation"]["voxel_deviation"]


def figure_three_levels(df: pd.DataFrame, path: Path):
    cases = [CASE_CLEAN, CASE_HARD]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), dpi=DPI)

    for row, case in enumerate(cases):
        rec = df[df.case_name == case].iloc[0]
        vox = _load_voxels(case)
        centers = np.array(vox["centers"])
        mean_abs = np.array(vox["mean_abs"], dtype=float)

        # Ebene 1: global + je Region
        ax = axes[row, 0]
        names = ["global", "Oberseite", "Flanke A", "Flanke B"]
        vals = [rec.global_mean_abs, rec.region_top_mean_abs,
                rec.region_flank_a_mean_abs, rec.region_flank_b_mean_abs]
        colors = [C_REF, "#B0BEC5", "#1976D2", "#FF9800"]
        bars = ax.bar(names, vals, color=colors, zorder=3)
        ax.axhline(TOLERANCE_MM, color=C_BAD, linestyle="--", linewidth=1.4,
                   zorder=4, label="0.25 mm")
        for b, v in zip(bars, vals):
            # Bei den sauberen Faellen liegen alle Werte unter 1 µm; mit drei
            # Nachkommastellen stuende ueberall 0.000.
            txt = f"{v:.3f}" if v >= 5e-4 else f"{v:.2e}"
            ax.text(b.get_x() + b.get_width() / 2, v, txt, ha="center",
                    va="bottom", fontsize=8, color=C_REF)
        ax.set_ylim(0, max(max(vals) * 1.35, TOLERANCE_MM * 1.4))
        _style(ax, ylabel="mittlerer |Abstand| (mm)",
               title=f"{case} — Ebene 1: global")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", labelrotation=15)

        # Ebene 2: Voxel, Draufsicht XY
        ax = axes[row, 1]
        vmax = max(TOLERANCE_MM, float(np.nanpercentile(mean_abs, 99)))
        sc = ax.scatter(centers[:, 0], centers[:, 1], c=mean_abs, cmap="viridis",
                        s=26, marker="s", vmin=0, vmax=vmax, linewidths=0)
        n_out = int((mean_abs > TOLERANCE_MM).sum())
        plt.colorbar(sc, ax=ax, label="|Abstand| je Voxel (mm)")
        _style(ax, xlabel="X — Naht-Längsrichtung (mm)", ylabel="Y (mm)",
               title=f"Ebene 2: Voxel ({n_out}/{len(mean_abs)} über Toleranz)")

        # Ebene 3: Merkmale
        ax = axes[row, 2]
        # Der Spalt-Restfehler ist nur dort ein Fehler, wo das Sollmodell die
        # Geometrie vollstaendig beschreibt. Bei ry/rz variiert der Spalt
        # entlang der Naht; der Wert steht dann fuer echte Geometrie.
        gap_label = ("Spalt-Restfehler" if rec.gap_model_complete
                     else "Spalt vs. Sollmodell\n(unvollständig: ry/rz)")
        feats = [
            ("Flankenwinkel A − 45°", rec.flank_a_angle_deg - 45, "°", True),
            ("Flankenwinkel B − 45°", rec.flank_b_angle_deg - 45, "°", True),
            ("Asymmetrie", rec.flank_asymmetry_deg, "°", True),
            ("Kantenversatz", rec.edge_offset_mm, "mm", True),
            ("Verkippung gesamt", rec.tilt_total_deg, "°", True),
            (gap_label, rec.gap_residual_mm, "mm", bool(rec.gap_model_complete)),
        ]
        ypos = np.arange(len(feats))[::-1]
        vals = [f[1] for f in feats]
        cols = [C_BAD if (abs(v) > 0.25 and judge) else "#1976D2"
                for _, v, _, judge in feats]
        ax.barh(ypos, vals, color=cols, zorder=3, height=0.6)
        ax.axvline(0, color=C_REF, linewidth=1.0, zorder=4)
        ax.set_yticks(ypos)
        ax.set_yticklabels([f[0] for f in feats], fontsize=9)
        for y, (name, v, unit, _) in zip(ypos, feats):
            ax.text(v, y, f"  {v:+.3f} {unit}", va="center", fontsize=8,
                    ha="left" if v >= 0 else "right", color=C_REF)
        span = max(abs(min(vals)), abs(max(vals)), 0.1) * 1.9
        ax.set_xlim(-span, span)
        _style(ax, title="Ebene 3: Merkmale")

    fig.suptitle("Drei Auswertungsebenen: sauberer Fall (oben) gegen schweren Fall "
                 "(unten)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── Textzusammenfassung ───────────────────────────────────────────────

def summarise(df: pd.DataFrame) -> None:
    def line(t):
        print(t)

    line("=" * 72)
    line(f"SYNTHETISCHE VALIDIERUNG — {len(df)} Fälle, "
         f"{'alle verankert' if df.anchored.all() else 'NICHT alle verankert'}")
    line("=" * 72)

    full = df[df.gap_model_complete]
    line(f"\n[1] Restfehler der Spaltmessung — {len(full)} von {len(df)} Fällen, "
         f"für die das Sollmodell vollständig ist")
    line("    (ry/rz erzeugen einen längs veränderlichen Spalt; dort ist die "
         "mittlere\n     Breite kein Vergleichswert, siehe [1b])")
    g = full.groupby("error_class").gap_residual_mm.agg(
        n="count", median="median", mean="mean",
        max_abs=lambda s: s.abs().max())
    line(g.reindex(CLASS_ORDER).dropna(how="all").round(5).to_string())
    within = (full.gap_residual_mm.abs() <= TOLERANCE_MM).sum()
    line(f"\n  innerhalb ±0.25 mm: {within}/{len(full)}")
    line(f"  Median:             {full.gap_residual_mm.median():+.5f} mm")
    line(f"  Spannweite:         {full.gap_residual_mm.min():+.5f} … "
         f"{full.gap_residual_mm.max():+.5f} mm")
    out = full[full.gap_residual_mm.abs() > TOLERANCE_MM]
    if len(out):
        line("\n  Fälle ausserhalb der Toleranz:")
        line(out[["case_name", "category", "gap_residual_mm", "bin_coverage",
                  "flank_asymmetry_deg"]].round(4).to_string(index=False))

    pure_rz = df[(df.rz_deg != 0) & (df.ry_deg == 0)]
    mixed = df[(df.rz_deg != 0) & (df.ry_deg != 0)]
    line(f"\n[1b] Spaltkeil: gemessene Steigung gegen −tan(rz)")
    line(f"  nur rz (n={len(pure_rz)}):      max|Fehler| "
         f"{pure_rz.wedge_slope_err.abs().max():.6f} mm/mm "
         f"= {np.degrees(np.arctan(pure_rz.wedge_slope_err.abs().max())):.4f}°")
    line(f"  rz und ry (n={len(mixed)}):   max|Fehler| "
         f"{mixed.wedge_slope_err.abs().max():.6f} mm/mm — hier fehlt der "
         f"ry-Beitrag im Sollmodell,\n"
         f"                       der Rest ist daher keine Messabweichung")

    line("\n[2] Merkmale gegen Ground Truth")
    tz = df[df.category == "translation_z"]
    line(f"  Kantenversatz (T_Z, n={len(tz)}): Bias "
         f"{tz.edge_offset_err_mm.mean():+.5f} mm, "
         f"Streuung {tz.edge_offset_err_mm.std():.6f} mm")
    rx = df[df.rx_deg != 0]
    slope, icpt = np.polyfit(rx.rx_deg, rx.angle_sum_err, 1)
    line(f"  Winkelsumme vs. rx (n={len(rx)}): Steigung {slope:.5f}, "
         f"Achsenabschnitt {icpt:+.5f}°, max|Rest| "
         f"{np.abs(rx.angle_sum_err - rx.rx_deg).max():.4f}°")
    tilt_s, tilt_i = np.polyfit(df.rx_deg, -df.tilt_across_gap_deg, 1)
    line(f"  Querverkippung vs. rx (n={len(df)}): Steigung {tilt_s:.5f}, "
         f"Achsenabschnitt {tilt_i:+.5f}°")
    line(f"  Flankenwinkel je Seite: {df.flank_a_angle_deg.min():.3f}°"
         f"…{df.flank_a_angle_deg.max():.3f}° (A), "
         f"{df.flank_b_angle_deg.min():.3f}°…{df.flank_b_angle_deg.max():.3f}° (B)")
    tr = df[df.error_class == "Translation"]
    line(f"    davon reine Translation: {tr.flank_a_angle_deg.min():.3f}°"
         f"…{tr.flank_a_angle_deg.max():.3f}°")

    line("\n[3] Bin-Abdeckung und Fit-Güte")
    inc = df[df.n_bins_valid < df.n_bins]
    line(f"  Fälle mit unvollständiger Abdeckung: {len(inc)}")
    if len(inc):
        line(inc[["case_name", "n_bins_valid", "n_bins"]].to_string(index=False))
    line(f"  min R² über alle Flanken: {df.fit_quality_min_r2.min():.6f}")
    line(f"  Referenzebene inlier_ratio: min {df.ref_plane_inlier_ratio.min():.4f}, "
         f"rms max {df.ref_plane_rms_mm.max():.4f} mm")

    line("\n[4] Zwei Beispielfälle")
    cols = ["case_name", "global_mean_abs", "global_in_tol_rate",
            "vox_frac_out_of_tol", "gap_residual_mm", "flank_asymmetry_deg",
            "edge_offset_mm", "tilt_total_deg"]
    line(df[df.case_name.isin([CASE_CLEAN, CASE_HARD])][cols].round(4).to_string(index=False))


def main() -> int:
    force_utf8_output()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    df = load_results(OUTPUTS, META)
    if df.empty:
        print("Keine Ergebnisse gefunden.", file=sys.stderr)
        return 1
    if df.attrs["missing"]:
        print(f"WARNUNG: {len(df.attrs['missing'])} Fälle ohne Report: "
              f"{df.attrs['missing']}", file=sys.stderr)

    df["angle_sum_err"] = ((df.flank_a_angle_deg - NOMINAL_FLANK_ANGLE_DEG)
                           + (df.flank_b_angle_deg - NOMINAL_FLANK_ANGLE_DEG))

    figure_error_distribution(df, FIGDIR / "01_restfehler_nach_klasse.png")
    figure_features(df, FIGDIR / "02_merkmale_vs_groundtruth.png")
    figure_three_levels(df, FIGDIR / "03_drei_ebenen_beispielfaelle.png")
    df.to_csv(FIGDIR / "synthetic_validation_features.csv", index=False)

    summarise(df)
    print(f"\nPlots und Merkmalstabelle: {FIGDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
