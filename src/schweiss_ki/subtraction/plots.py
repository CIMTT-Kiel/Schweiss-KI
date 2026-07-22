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

# Annotationsebene des Querschnitts – bewusst KEINE neuen kategorialen Farben.
# Referenzebene, Fitband-Grenzen und Wurzeltiefe sind Bezugsgrößen, keine
# Datenreihen; sie tragen neutrale Ink-Tokens, damit die Label-Farben
# eindeutig den Flanken zugeordnet bleiben.
COLOR_REFERENCE = "#37474F"   # dunkelgrau – Referenzebene (der Anker)
COLOR_GUIDE = "#B0BEC5"       # hellgrau – Fitband-Grenzen, Hilfslinien


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

def _plane_z_at(plane: dict, seam_val: float, gap_vals: np.ndarray,
                seam_axis: int, gap_axis: int, vertical_axis: int,
                depth: float = 0.0) -> np.ndarray:
    """Z-Höhe einer Ebene an gegebenen gap-Positionen, optional in Tiefe `depth`.

    Löst n·p + d = -depth nach der Vertikalkomponente auf. `depth` wird
    senkrecht zur Ebene gemessen (so wie GapProfile es definiert), nicht
    vertikal – bei geneigter Ebene ist das ein Unterschied.
    """
    n = np.asarray(plane["normal"], dtype=float)
    gap_vals = np.atleast_1d(np.asarray(gap_vals, dtype=float))
    fixed = n[seam_axis] * seam_val + n[gap_axis] * gap_vals + float(plane["d"])
    return (-fixed - depth) / n[vertical_axis]


def _bins_in_window(profile: dict, x_min: float, x_max: float) -> np.ndarray:
    """Indizes der Naht-Bins, deren Zentrum im Querschnittsfenster liegt."""
    centers = np.asarray(profile.get("seam_axis_centers", []), dtype=float)
    if centers.size == 0:
        return np.empty(0, dtype=int)
    return np.where((centers >= x_min) & (centers < x_max))[0]


def plot_cross_section(
    model: "WeldVolumeModel",
    x_min: float,
    x_max: float,
    show_anchored: bool = True,
    show_extrapolation: bool = False,
    labels_to_show: Sequence[int] = (1, 2, 3, 4),
    figsize: tuple = (9, 6.5),
) -> plt.Figure:
    """Querschnitt YZ für einen X-Bereich entlang der Naht.

    Visuelle Kontrolle der verankerten Spaltmessung. Gezeichnet werden die
    im DeviationReport GESPEICHERTEN Fits – nicht neu gerechnete. Ein
    zweiter Fit an dieser Stelle würde Abweichungen zwischen Bild und
    Messwert verstecken.

    Elemente (show_anchored=True):
        - Rohpunkte je Label, im Hintergrund
        - Referenzebene bei d = 0, mit inlier_ratio und rms in der Legende:
          der Anker, an dem die gesamte Messung hängt
        - die zwei Flankengeraden, gezeichnet NUR über ihr tatsächliches
          Fitband – die Bandgrenzen sind damit direkt ablesbar
        - Fitband-Grenzen (d_min bzw. P95-Schnitt) als Hilfslinien
        - Wurzeltiefe d_root: dort sitzt der bekannte systematische Offset
          von ~0.019 mm. Wer den Wert im Bild sieht, kann bei realen Scans
          beurteilen, ob P95 dort sinnvoll liegt oder die Flankenabdeckung
          ein anderes Quantil verlangt.

    show_extrapolation blendet zusätzlich die alte z=0-Methode ein
    (Default aus), solange beide Verfahren parallel laufen.

    Diagnostisches Werkzeug: Gratbildung, Wurzeldurchhang, Heftnähte und
    ungewöhnliche Punktverteilungen sind hier sofort sichtbar – ein r² < 1
    an einer Flanke lässt sich gegen die Rohpunkte prüfen.
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
        # Rohpunkte bewusst zurückgenommen und im Hintergrund: sie sind die
        # Prüfgrundlage für die Fits, nicht die Aussage des Plots.
        ax.scatter(
            pts[lmask, 1], pts[lmask, 2],
            c=LABEL_COLORS[label_id], s=8, alpha=0.35, linewidths=0, zorder=2,
            label=f"{LABEL_NAMES[label_id]} ({int(lmask.sum())})",
        )

    # ── Verankerte Auswertung: gespeicherte Fits zeichnen ─────────────
    # Bewusst NICHT lokal neu fitten – der Plot soll zeigen, was GapProfile
    # tatsächlich gerechnet hat. Ein zweiter Fit hier würde Abweichungen
    # zwischen Plot und Messwert verstecken, also genau das kaschieren, wofür
    # die visuelle Kontrolle da ist.
    profile = None
    if model.subtraction_report is not None:
        profile = getattr(model.subtraction_report.deviation, "gap_profile", None)

    drawn_anchored = False
    if show_anchored and profile and profile.get("anchored"):
        s_ax = int(profile.get("seam_axis", 0))
        g_ax = int(profile.get("gap_axis", 1))
        v_ax = int(profile.get("vertical_axis", 2))
        ref = profile.get("reference_plane")
        bins = _bins_in_window(profile, x_min, x_max)
        seam_c = 0.5 * (x_min + x_max)
        y_span = np.linspace(pts[mask, g_ax].min(), pts[mask, g_ax].max(), 2)

        if ref is not None:
            # Referenzebene bei d = 0 – der Anker, an dem alles hängt.
            z_ref = _plane_z_at(ref, seam_c, y_span, s_ax, g_ax, v_ax, depth=0.0)
            ax.plot(
                y_span, z_ref, "-", color=COLOR_REFERENCE, linewidth=2.0, zorder=6,
                label=(f"Referenzebene d=0 (inlier {ref['inlier_ratio']:.3f}, "
                       f"rms {ref['rms_mm']:.3f} mm)"),
            )

        for label_id, key in ((1, "flank_a_profile"), (2, "flank_b_profile")):
            prof = profile.get(key)
            if prof is None or len(bins) == 0:
                continue
            q0 = np.asarray(prof["q0"], dtype=float)
            sl = np.asarray(prof["slope"], dtype=float)
            d_lo = np.asarray(prof["d_lo"], dtype=float)
            d_hi = np.asarray(prof["d_hi"], dtype=float)
            r2 = np.asarray(prof["r2"], dtype=float)
            for bi in bins:
                if not np.isfinite(q0[bi]):
                    continue
                # Nur über das tatsächlich gefittete Tiefenband zeichnen –
                # so ist im Bild ablesbar, wo gefittet wurde und wo nicht.
                dd = np.linspace(d_lo[bi], d_hi[bi], 40)
                qq = q0[bi] + sl[bi] * dd
                zz = np.array([
                    _plane_z_at(ref, seam_c, q, s_ax, g_ax, v_ax, depth=d)[0]
                    for q, d in zip(qq, dd)
                ])
                first = bi == bins[0]
                ax.plot(
                    qq, zz, "-", color=LABEL_COLORS[label_id], linewidth=2.0, zorder=8,
                    label=(f"{LABEL_NAMES[label_id]} Fit "
                           f"(α={np.degrees(np.arctan(abs(sl[bi]))):.2f}°, "
                           f"r²={r2[bi]:.4f})") if first else None,
                )
            drawn_anchored = True

        # Obere Fitband-Grenze als Hilfslinie – das ist ein globaler
        # Parameter (flank_depth_min) und gilt fuer beide Flanken.
        #
        # Die UNTERE Grenze wird bewusst NICHT als gemeinsame Linie gezogen:
        # sie ist je Flanke verschieden (P95 der jeweiligen Abdeckung), und
        # ein Aggregat waere bei asymmetrischer Abdeckung irrefuehrend. Wo
        # jede Flanke endet, zeigt ihr eigenes Linienende bereits exakt.
        if ref is not None and len(bins):
            d_lo_all = [np.asarray(profile[k]["d_lo"], dtype=float)[bins]
                        for k in ("flank_a_profile", "flank_b_profile")]
            band_lo = np.nanmin(np.concatenate(d_lo_all)) if len(d_lo_all) else np.nan
            if np.isfinite(band_lo):
                ax.plot(
                    y_span, _plane_z_at(ref, seam_c, y_span, s_ax, g_ax, v_ax, depth=band_lo),
                    ":", color=COLOR_GUIDE, linewidth=1.2, zorder=4,
                    label=f"Fitband-Obergrenze (d={band_lo:.2f} mm)",
                )
            d_root = np.asarray(profile.get("d_root", []), dtype=float)
            if d_root.size and np.isfinite(d_root[bins]).any():
                dr = float(np.nanmean(d_root[bins]))
                gr = np.asarray(profile.get("gap_root_widths", []), dtype=float)
                gr_txt = (f", Spalt {np.nanmean(gr[bins]):.3f} mm"
                          if gr.size and np.isfinite(gr[bins]).any() else "")
                ax.plot(
                    y_span, _plane_z_at(ref, seam_c, y_span, s_ax, g_ax, v_ax, depth=dr),
                    "--", color=COLOR_REFERENCE, linewidth=1.5, alpha=0.8, zorder=5,
                    label=f"Wurzeltiefe d_root={dr:.3f} mm{gr_txt}",
                )

    # ── Legacy: lokale Extrapolation auf z = 0 ────────────────────────
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
                color=LABEL_COLORS[label_id], alpha=0.35, linewidth=1.2, zorder=3,
                label=f"{LABEL_NAMES[label_id]} z=0-Extrapolation (y₀={y0:+.3f})",
            )
            ax.scatter([y0], [0], color=LABEL_COLORS[label_id], marker="x",
                       s=80, alpha=0.5, zorder=7)
        ax.axhline(0, color=COLOR_GUIDE, linestyle=":", alpha=0.6,
                   label="z = 0 (alte Bezugsebene)")

    # Fehlende Verankerung sichtbar machen, nicht nur loggen – und die
    # Ursachen auseinanderhalten. Ein alter Report ohne gap_profile darf
    # nicht wie eine fehlgeschlagene Verankerung aussehen, sonst debuggt
    # man spaeter die Verankerung, obwohl nur die Daten fehlen.
    if show_anchored and not drawn_anchored:
        if profile is None:
            grund = ("Profil nicht im Report – vor Einfuehrung der\n"
                     "Serialisierung geschrieben. Batch neu laufen lassen.")
        elif not profile.get("anchored"):
            ref = profile.get("reference_plane") or {}
            ir = ref.get("inlier_ratio")
            grund = ("Verankerung fehlgeschlagen"
                     + (f" (inlier_ratio {ir:.3f})" if ir is not None else ""))
        else:
            grund = f"kein Bin-Zentrum in X ∈ [{x_min:.1f}, {x_max:.1f}]"
        logger.warning(f"  plot_cross_section: keine verankerte Auswertung – {grund}")
        ax.text(
            0.02, 0.02, f"⚠ keine verankerte Auswertung\n{grund}",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
            color=COLOR_REFERENCE,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF3E0",
                      edgecolor=COLOR_GUIDE, alpha=0.95),
            zorder=20,
        )
    # ── Ausschnitt auf die Naht begrenzen ─────────────────────────────
    # Die Deckflächen reichen über die volle Blechbreite (±50 mm), die Naht
    # ist nur wenige mm breit. Ohne Zoom quetscht aspect="equal" den
    # interessanten Bereich zu einem Splitter zusammen. Gezoomt wird auf die
    # Flanken, damit Flankenwinkel und Spalt ablesbar bleiben.
    flank_mask = mask & np.isin(model.labels, (1, 2))
    if flank_mask.any():
        f_lo, f_hi = pts[flank_mask, 1].min(), pts[flank_mask, 1].max()
        pad = max(0.25 * (f_hi - f_lo), 1.0)
        ax.set_xlim(f_lo - pad, f_hi + pad)
        z_lo, z_hi = pts[flank_mask, 2].min(), pts[flank_mask, 2].max()
        z_pad = max(0.15 * (z_hi - z_lo), 0.5)
        ax.set_ylim(z_lo - z_pad, z_hi + z_pad)

    ax.set_xlabel("Y – Spalt-Querrichtung (mm)")
    ax.set_ylabel("Z – Tiefe (mm)")
    ax.set_aspect("equal")   # Flankenwinkel sollen unverzerrt ablesbar sein
    ax.set_title(f"Querschnitt X ∈ [{x_min:+.1f}, {x_max:+.1f}] – {model.model_id}")
    # Legende neben die Achse: sie ist mit Fit-Kennwerten lang und würde
    # innerhalb der Achse die Naht verdecken.
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        framealpha=0.95, fontsize=8, borderaxespad=0.0,
    )
    ax.grid(alpha=0.25, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
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