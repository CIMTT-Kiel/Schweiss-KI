"""Plotly-Figuren, datengetrieben aus dem gewählten Fall.

Farbskala identisch zu den Präsentationsbildern (`presentation_figures.py`):
**blau = Material fehlt · neutral = in Toleranz · rot = steht über** — ein
divergierender Verlauf mit neutralem Grau in der Mitte, kein eigener Farbton
fürs Toleranzband. Achsengrenzen und Farbgrenze kommen aus den Daten des Falls,
nichts ist fest gesetzt, damit reale Scans mit anderer Ausdehnung nicht brechen.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from schweiss_ki.analysis.deviation_field import TOLERANCE_MM

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


def build_3d(cad_pts: np.ndarray, scan_pts: np.ndarray, signed: np.ndarray,
             *, show_cad: bool = True, z_exagg: float = 1.0) -> go.Figure:
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
