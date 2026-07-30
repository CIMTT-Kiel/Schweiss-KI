"""Dash-App: Layout und Callbacks.

Iterativer Aufbau. Stand: Fallauswahl + 3D-Punktwolke funktionsfähig; die
weiteren Reiter (Abweichungskarte, Drei Ebenen, Gap-Profil) sind angelegt und
werden in den nächsten Iterationen gefüllt.
"""
from __future__ import annotations

from pathlib import Path

import dash
from dash import Input, Output, dcc, html
import plotly.graph_objects as go

from . import data
from .figures import INK, build_3d

POINT_OPTIONS = [
    {"label": "20 000 — flott drehen", "value": 20000},
    {"label": "50 000", "value": 50000},
    {"label": "80 000", "value": 80000},
    {"label": "volle Dichte — langsam", "value": 0},
]
CAD_DISPLAY_N = 25000

_SOURCE_BADGE = {
    "synthetic": ("Synthetische Daten", "#eb6834"),
    "real": ("Reale Messung", "#0ca30c"),
}
_LABEL = {"color": INK, "fontWeight": 600, "fontSize": "13px",
          "marginRight": "6px"}
_BOX = {"display": "flex", "flexDirection": "column", "gap": "3px"}


def _placeholder(text: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=text, showarrow=False,
                       font=dict(size=15, color=INK), x=0.5, y=0.5,
                       xref="paper", yref="paper")
    fig.update_layout(paper_bgcolor="#fff", plot_bgcolor="#fff",
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      margin=dict(l=0, r=0, t=0, b=0))
    return fig


def build_app(outputs_dir: Path = data.DEFAULT_OUTPUTS) -> dash.Dash:
    outputs_dir = Path(outputs_dir)
    cases = data.discover_cases(outputs_dir)
    app = dash.Dash(__name__, title="Abweichungs-Dashboard")

    controls = html.Div([
        html.Div([html.Label("Fall", style=_LABEL),
                  dcc.Dropdown(id="case", options=[{"label": c, "value": c}
                                                   for c in cases],
                               value=cases[0] if cases else None,
                               clearable=False,
                               style={"width": "260px"})], style=_BOX),
        html.Div([html.Label("Punkte", style=_LABEL),
                  dcc.Dropdown(id="npoints", options=POINT_OPTIONS,
                               value=50000, clearable=False,
                               style={"width": "220px"})], style=_BOX),
        html.Div([html.Label("CAD-Ideal", style=_LABEL),
                  dcc.Checklist(id="showcad",
                                options=[{"label": " anzeigen", "value": "on"}],
                                value=["on"],
                                style={"color": INK, "fontSize": "13px"})],
                 style=_BOX),
        html.Div(id="badge", style={"marginLeft": "auto",
                                    "alignSelf": "center"}),
    ], style={"display": "flex", "gap": "22px", "alignItems": "flex-end",
              "flexWrap": "wrap", "padding": "4px 6px 12px"})

    app.layout = html.Div([
        html.H2("Abweichungsanalyse — Auswertungs-Dashboard",
                style={"color": INK, "margin": "6px 6px 2px",
                       "fontFamily": "system-ui, sans-serif"}),
        html.Div(id="caseinfo", style={"color": INK, "fontSize": "13px",
                                       "margin": "0 6px 8px"}),
        controls,
        dcc.Tabs(id="tabs", value="3d", children=[
            dcc.Tab(label="3D-Punktwolke", value="3d"),
            dcc.Tab(label="Abweichungskarte", value="map"),
            dcc.Tab(label="Drei Ebenen", value="levels"),
            dcc.Tab(label="Gap-Profil", value="gap"),
        ]),
        dcc.Loading(dcc.Graph(id="view", style={"height": "76vh"},
                              config={"displaylogo": False}),
                    type="circle"),
    ], style={"maxWidth": "1500px", "margin": "0 auto",
              "fontFamily": "system-ui, sans-serif"})

    _register(app, outputs_dir)
    return app


def _register(app: dash.Dash, outputs_dir: Path) -> None:
    od = str(outputs_dir)

    @app.callback(
        Output("view", "figure"), Output("badge", "children"),
        Output("badge", "style"), Output("caseinfo", "children"),
        Input("tabs", "value"), Input("case", "value"),
        Input("npoints", "value"), Input("showcad", "value"))
    def _update(tab, case, npoints, showcad):
        if not case:
            return _placeholder("Keine Fälle in data/outputs gefunden."), \
                "", {"display": "none"}, ""

        src = data.source_type(od, case)
        text, color = _SOURCE_BADGE.get(src, (f"Quelle: {src}", "#898781"))
        badge_style = {"marginLeft": "auto", "alignSelf": "center",
                       "background": color, "color": "#fff",
                       "padding": "4px 12px", "borderRadius": "999px",
                       "fontSize": "12px", "fontWeight": 700}

        field = data.case_field(outputs_dir, case)
        info = (f"in Toleranz: {field.in_tolerance_rate * 100:.1f} %   ·   "
                f"{len(field.points):,} Punkte   ·   Quelle: {src}")

        if tab == "3d":
            scan_pts, signed = data.downsampled_cloud(field, npoints)
            cad = (data.cad_top_points(outputs_dir, CAD_DISPLAY_N)
                   if showcad else [])
            import numpy as np
            fig = build_3d(np.asarray(cad) if len(cad) else np.empty((0, 3)),
                           scan_pts, signed, show_cad=bool(showcad))
        else:
            fig = _placeholder("Diese Ansicht kommt in der nächsten Iteration.")

        return fig, text, badge_style, info
