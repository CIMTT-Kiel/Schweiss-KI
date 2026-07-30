"""Plotly-Figuren, datengetrieben aus dem gewählten Fall.

Farbskala identisch zu den Präsentationsbildern (`presentation_figures.py`):
**blau = Material fehlt · neutral = in Toleranz · rot = steht über** — ein
divergierender Verlauf mit neutralem Grau in der Mitte, kein eigener Farbton
fürs Toleranzband. Achsengrenzen und Farbgrenze kommen aus den Daten des Falls,
nichts ist fest gesetzt, damit reale Scans mit anderer Ausdehnung nicht brechen.
"""
from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from schweiss_ki.analysis.deviation_field import TOLERANCE_MM
from schweiss_ki.analysis.synthetic_validation import extract_features

# Segmentierungslabel → Farbe/Name, konsistent zu subtraction/plots.py.
LABEL_STYLE = {0: ("#b0bec5", "Oberseite/Hintergrund"),
               1: ("#1976d2", "Flanke A"), 2: ("#ff9800", "Flanke B"),
               3: ("#43a047", "Spalt"), 4: ("#9c27b0", "Sub-Gap")}
REGION_NAME = {"0": "Oberseite", "1": "Flanke A", "2": "Flanke B"}

# Diverging blau → neutral → rot, wie DEV_CMAP in presentation_figures.py.
DEV_COLORSCALE = [
    [0.0, "#2a78d6"], [0.25, "#a9c6ea"], [0.5, "#e8e7e2"],
    [0.75, "#eeab9f"], [1.0, "#e34948"],
]
C_CAD = "#9aa6b0"

PLOT_BG = "#ffffff"
INK = "#141a1f"


def robust_vmax(signed: np.ndarray, tol: float = TOLERANCE_MM) -> float:
    """Symmetrische Farbgrenze aus den Daten — 99. Perzentil, mind. 1.5·Toleranz."""
    finite = signed[np.isfinite(signed)]
    if finite.size == 0:
        return tol * 1.5
    return float(max(np.percentile(np.abs(finite), 99), tol * 1.5))


def _colorbar(vmax: float) -> dict:
    return dict(
        title=dict(text="Abstand (mm)", side="right", font=dict(color=INK)),
        tickfont=dict(color=INK), thickness=16, len=0.7,
        tickvals=[-vmax, -TOLERANCE_MM, 0, TOLERANCE_MM, vmax],
    )


def _aspect(scan_pts: np.ndarray, cad_pts: np.ndarray, z_exagg: float) -> dict:
    """Achsenverhältnis aus den Datenausdehnungen, Z optional überhöht.

    z_exagg=1 → echte Proportionen (wie aspectmode='data'). Bei flachen
    Bauteilen (Z ≪ X) machen grössere Werte Höhen-/Kippfehler geometrisch
    sichtbar, ohne die Messwerte zu verändern — nur die Darstellung.
    """
    pts = scan_pts if not len(cad_pts) else np.vstack([scan_pts, cad_pts])
    rng = np.ptp(pts, axis=0).astype(float)
    rng[rng < 1e-6] = 1.0
    ar = np.array([rng[0], rng[1], rng[2] * z_exagg])
    ar /= ar.max()
    return dict(x=float(ar[0]), y=float(ar[1]), z=float(ar[2]))


# Achsenkreuz — eigene Farben, klar getrennt von der blau/rot-Abweichungsskala.
TRIAD_COLORS = {"X": "#e07b00", "Y": "#1a9e5f", "Z": "#7048b6"}


def _add_triad(fig: go.Figure, extent_pts: np.ndarray) -> None:
    """Mitrotierendes X/Y/Z-Achsenkreuz mit Pfeilen in einer Ecke der Szene.

    So sieht man beim Drehen sofort, um welche Achse gedreht wird. Liegt in
    Datenkoordinaten, wird also von der Z-Streckung konsistent mitverzerrt.
    """
    if not len(extent_pts):
        return
    mn, mx = extent_pts.min(0), extent_pts.max(0)
    rng = np.where((mx - mn) < 1e-6, 1.0, mx - mn)
    L = 0.16 * float(rng.max())                 # gleiche Datenlänge je Achse
    o = np.zeros(3)                             # am Koordinatenursprung (0,0,0)
    for axis, vec in (("X", (L, 0, 0)), ("Y", (0, L, 0)), ("Z", (0, 0, L))):
        col = TRIAD_COLORS[axis]
        tip = o + np.array(vec)
        fig.add_trace(go.Scatter3d(
            x=[o[0], tip[0]], y=[o[1], tip[1]], z=[o[2], tip[2]], mode="lines",
            line=dict(color=col, width=6), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Cone(
            x=[tip[0]], y=[tip[1]], z=[tip[2]], u=[vec[0]], v=[vec[1]],
            w=[vec[2]], sizemode="absolute", sizeref=L * 0.32, anchor="tip",
            showscale=False, colorscale=[[0, col], [1, col]], hoverinfo="skip"))
        lab = o + 1.16 * np.array(vec)
        fig.add_trace(go.Scatter3d(
            x=[lab[0]], y=[lab[1]], z=[lab[2]], mode="text", text=[axis],
            textfont=dict(color=col, size=16, family="system-ui"),
            showlegend=False, hoverinfo="skip"))


def build_3d(cad_pts: np.ndarray, scan_pts: np.ndarray, signed: np.ndarray,
             *, show_cad: bool = True, z_exagg: float = 1.0,
             show_triad: bool = True) -> go.Figure:
    """CAD-Ideal (grau) und Fall (nach signiertem Abstand eingefärbt), drehbar."""
    vmax = robust_vmax(signed)
    fig = go.Figure()

    if show_cad and len(cad_pts):
        fig.add_trace(go.Scatter3d(
            x=cad_pts[:, 0], y=cad_pts[:, 1], z=cad_pts[:, 2], mode="markers",
            marker=dict(size=1.5, color=C_CAD, opacity=0.30),
            name="CAD-Ideal", hoverinfo="skip"))

    fig.add_trace(go.Scatter3d(
        x=scan_pts[:, 0], y=scan_pts[:, 1], z=scan_pts[:, 2], mode="markers",
        marker=dict(size=2.0, color=signed, colorscale=DEV_COLORSCALE,
                    cmin=-vmax, cmax=vmax, opacity=0.95,
                    colorbar=_colorbar(vmax)),
        name="Scan",
        hovertemplate="X %{x:.1f} · Y %{y:.1f} · Z %{z:.1f} mm<br>"
                      "Abstand %{marker.color:+.3f} mm<extra></extra>"))

    if show_triad:
        extent = (np.vstack([scan_pts, cad_pts])
                  if show_cad and len(cad_pts) else scan_pts)
        _add_triad(fig, extent)

    z_title = "Z — Tiefe (mm)" + (f"  ·  {z_exagg:g}× überhöht"
                                  if z_exagg > 1 else "")
    fig.update_layout(
        scene=dict(
            aspectmode="manual",
            aspectratio=_aspect(scan_pts, cad_pts, z_exagg),
            xaxis=dict(title="X — Naht-Längs (mm)", color=INK,
                       backgroundcolor=PLOT_BG),
            yaxis=dict(title="Y — quer (mm)", color=INK,
                       backgroundcolor=PLOT_BG),
            zaxis=dict(title=z_title, color=INK, backgroundcolor=PLOT_BG),
        ),
        paper_bgcolor=PLOT_BG, font=dict(color=INK),
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
        uirevision="keep",   # Blickwinkel über Fallwechsel hinweg halten
    )
    return fig


def _finite(seq) -> np.ndarray:
    a = np.array([np.nan if v is None else v for v in seq], dtype=float)
    return a


def _base_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                      font=dict(color=INK), margin=dict(l=55, r=20, t=40, b=45))
    fig.update_xaxes(color=INK, gridcolor="#e1e0d9", zeroline=False)
    fig.update_yaxes(color=INK, gridcolor="#e1e0d9", zeroline=False)
    return fig


# ── Abweichungskarte (Draufsicht) ─────────────────────────────────────

def build_map(scan_pts: np.ndarray, signed: np.ndarray) -> go.Figure:
    """Draufsicht auf die XY-Ebene, gefärbt nach signiertem Abstand."""
    vmax = robust_vmax(signed)
    fig = go.Figure(go.Scattergl(
        x=scan_pts[:, 0], y=scan_pts[:, 1], mode="markers",
        marker=dict(size=3.5, color=signed, colorscale=DEV_COLORSCALE,
                    cmin=-vmax, cmax=vmax, colorbar=_colorbar(vmax)),
        hovertemplate="X %{x:.1f} · Y %{y:.1f} mm<br>"
                      "Abstand %{marker.color:+.3f} mm<extra></extra>"))
    _base_layout(fig)
    fig.update_xaxes(title="X — Naht-Längsrichtung (mm)")
    fig.update_yaxes(title="Y — quer (mm)", scaleanchor="x", scaleratio=1)
    return fig


# ── Drei Ebenen (global / Voxel / Merkmale) ───────────────────────────

def _feature_rows(report: dict, in_tol_rate: float) -> list[tuple[str, str]]:
    f = extract_features(report)
    wsum = ((f["flank_a_angle_deg"] - 45.0) + (f["flank_b_angle_deg"] - 45.0)
            if np.isfinite(f["flank_a_angle_deg"]) else math.nan)

    def fmt(v, unit, dec=2):
        return "—" if v is None or not np.isfinite(v) else f"{v:+.{dec}f} {unit}"

    return [
        ("in Toleranz", f"{in_tol_rate * 100:.1f} %"),
        ("Ø |Abstand| (global)", fmt(f["global_mean_abs"], "mm", 3).lstrip("+")),
        ("Spaltbreite @ d_root", fmt(f["gap_width_mm"], "mm", 3).lstrip("+")),
        ("d_root (Auswertetiefe)", fmt(f["d_root_mm"], "mm", 2).lstrip("+")),
        ("Winkelsumme (α_A+α_B−90°)", fmt(wsum, "°", 2)),
        ("Flankenasymmetrie", fmt(f["flank_asymmetry_deg"], "°", 2).lstrip("+")),
        ("Kantenversatz", fmt(f["edge_offset_mm"], "mm", 2)),
        ("Verkippung gesamt", fmt(f["tilt_total_deg"], "°", 2).lstrip("+")),
        ("Translation Tx / Ty / Tz",
         " / ".join(fmt(f[f"measured_t{a}_mm"], "", 2) for a in "xyz") + " mm"),
        ("Rotation Rx / Ry / Rz",
         " / ".join(fmt(f[f"measured_r{a}_deg"], "", 2) for a in "xyz") + " °"),
    ]


def build_levels(report: dict, in_tol_rate: float) -> go.Figure:
    """Global (Regionsabweichung) · Voxel (räumlich) · Merkmale (Tabelle)."""
    dev = report["deviation"]
    fig = make_subplots(
        rows=1, cols=3, column_widths=[0.24, 0.42, 0.34],
        specs=[[{"type": "indicator"}, {"type": "xy"}, {"type": "table"}]],
        subplot_titles=("① Global: Anteil in Toleranz",
                        "② Voxel: räumliche Verteilung",
                        "③ Merkmale"),
        horizontal_spacing=0.06)

    rate = in_tol_rate * 100
    fig.add_trace(go.Indicator(
        mode="gauge+number", value=rate,
        number=dict(suffix=" %", font=dict(size=42, color=INK)),
        gauge=dict(axis=dict(range=[0, 100], tickcolor=INK,
                             tickfont=dict(color=INK)),
                   bar=dict(color="#0ca30c"), bgcolor="#eceff1",
                   borderwidth=0)), 1, 1)

    vox = dev.get("voxel_deviation", {})
    centers = _finite(vox.get("centers", [])).reshape(-1, 3) if vox.get("centers") \
        else np.empty((0, 3))
    ms = _finite(vox.get("mean_signed", []))
    if centers.size:
        vmaxv = robust_vmax(ms)
        fig.add_trace(go.Scattergl(
            x=centers[:, 0], y=centers[:, 1], mode="markers",
            marker=dict(size=7, symbol="square", color=ms,
                        colorscale=DEV_COLORSCALE, cmin=-vmaxv, cmax=vmaxv,
                        colorbar=dict(title="mm", thickness=12, len=0.5,
                                      x=0.63)),
            hovertemplate="X %{x:.0f} · Y %{y:.0f}<br>%{marker.color:+.2f} mm"
                          "<extra></extra>"), 1, 2)

    rows = _feature_rows(report, in_tol_rate)
    fig.add_trace(go.Table(
        columnwidth=[58, 42],
        header=dict(values=["<b>Merkmal</b>", "<b>Wert</b>"],
                    fill_color="#eceff1", align="left",
                    font=dict(color=INK, size=12)),
        cells=dict(values=[[r[0] for r in rows], [r[1] for r in rows]],
                   align="left", height=26,
                   font=dict(color=INK, size=12),
                   fill_color=[["#ffffff", "#f6f7f9"] * len(rows)])), 1, 3)

    _base_layout(fig)
    fig.update_xaxes(title="X (mm)", row=1, col=2)
    fig.update_yaxes(title="Y (mm)", scaleanchor="x2", scaleratio=1,
                     row=1, col=2)
    fig.update_layout(showlegend=False, margin=dict(l=45, r=20, t=55, b=45))
    return fig


# ── Gap-Profil / Querschnitt ──────────────────────────────────────────

def build_gap(report: dict, points: np.ndarray, labels: np.ndarray) -> go.Figure:
    """Spaltbreite entlang der Naht + Querschnitt (Flanken, Referenzebene)."""
    gp = report["deviation"].get("gap_profile", {})
    centers = _finite(gp.get("seam_axis_centers", []))
    widths = _finite(gp.get("gap_root_widths", []))

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.5, 0.5],
        subplot_titles=("Spaltbreite entlang der Naht",
                        "Querschnitt an einer Nahtstelle"),
        horizontal_spacing=0.09)

    if centers.size:
        fig.add_trace(go.Scatter(
            x=centers, y=widths, mode="lines+markers",
            line=dict(color="#1976d2"), marker=dict(size=5),
            hovertemplate="X %{x:.0f} mm<br>Spalt %{y:.3f} mm<extra></extra>"),
            1, 1)

    # Querschnitt: dünne X-Scheibe der echten Punkte, nach Label gefärbt,
    # plus die Referenzebene aus dem Report.
    xc = float(np.nanmedian(centers)) if centers.size else float(
        np.median(points[:, 0]))
    half = 6.0
    sl = np.abs(points[:, 0] - xc) < half
    for lab in (0, 1, 2):
        m = sl & (labels == lab)
        if not m.any():
            continue
        color, name = LABEL_STYLE[lab]
        fig.add_trace(go.Scattergl(
            x=points[m, 1], y=points[m, 2], mode="markers",
            marker=dict(size=3, color=color), name=name,
            hovertemplate=f"{name}<br>Y %{{x:.2f}} · Z %{{y:.2f}} mm"
                          "<extra></extra>"), 1, 2)

    ref = gp.get("reference_plane", {})
    n, d = ref.get("normal"), ref.get("d")
    if n and d is not None and abs(n[2]) > 1e-6 and sl.any():
        ys = np.array([points[sl, 1].min(), points[sl, 1].max()])
        zs = -(n[0] * xc + n[1] * ys + d) / n[2]
        fig.add_trace(go.Scatter(
            x=ys, y=zs, mode="lines",
            line=dict(color="#37474f", width=2.5, dash="dash"),
            name="Referenzebene"), 1, 2)

    _base_layout(fig)
    fig.update_xaxes(title="X — Naht-Längsrichtung (mm)", row=1, col=1)
    fig.update_yaxes(title="Spaltbreite (mm)", row=1, col=1)
    fig.update_xaxes(title=f"Y — quer (mm)   ·   Scheibe bei X≈{xc:.0f} mm",
                     row=1, col=2)
    fig.update_yaxes(title="Z — Tiefe (mm)", scaleanchor="x2", scaleratio=1,
                     row=1, col=2)
    fig.update_layout(legend=dict(x=0.55, y=0.98,
                                  bgcolor="rgba(255,255,255,0.7)"))
    return fig
