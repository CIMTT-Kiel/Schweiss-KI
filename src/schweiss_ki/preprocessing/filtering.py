"""
Filter-Steps für die Preprocessing-Pipeline.

Enthält alle Steps, die Punkte entfernen:
- StatisticalOutlierFilter: Entfernt statistisch abweichende Ausreißer
- RadiusOutlierFilter: Entfernt isolierte Punkte (Spritzer-Kandidaten)
"""
from __future__ import annotations

import open3d as o3d

from .base import PreprocessingStep


class StatisticalOutlierFilter(PreprocessingStep):
    """
    Statistischer Ausreißerfilter.

    Berechnet für jeden Punkt den mittleren Abstand zu seinen nb_neighbors
    nächsten Nachbarn. Punkte, deren mittlerer Abstand mehr als
    std_ratio Standardabweichungen vom globalen Mittel abweicht,
    werden entfernt.

    Empfohlene Startwerte:
        nb_neighbors=20, std_ratio=2.0

    Für stark verrauschte Scans std_ratio reduzieren (z.B. 1.5).
    Für CAD-Punktwolken kann dieser Step deaktiviert werden.
    """

    def __init__(
        self,
        nb_neighbors: int = 20,
        std_ratio: float = 2.0,
        enabled: bool = True,
    ):
        """
        Args:
            nb_neighbors: Anzahl betrachteter Nachbarn pro Punkt.
            std_ratio: Schwellwert in Vielfachen der Standardabweichung.
                       Niedrigere Werte = aggressiverer Filter.
            enabled: Step aktiv/inaktiv.
        """
        self._nb_neighbors = nb_neighbors
        self._std_ratio = std_ratio
        self._enabled = enabled

    @property
    def name(self) -> str:
        return "statistical_outlier_filter"

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _apply(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        pcd_clean, _ = pcd.remove_statistical_outlier(
            nb_neighbors=self._nb_neighbors,
            std_ratio=self._std_ratio,
        )
        return pcd_clean

    def get_params(self) -> dict:
        return {
            "nb_neighbors": self._nb_neighbors,
            "std_ratio": self._std_ratio,
        }


class RadiusOutlierFilter(PreprocessingStep):
    """
    Radius-basierter Ausreißerfilter.

    Entfernt Punkte, die innerhalb eines Radius von search_radius
    weniger als min_nb_points Nachbarn haben.

    Besonders geeignet zum Entfernen von isolierten Spatter-Punkten,
    die den statistischen Filter ggf. nicht triggern.

    Empfohlene Startwerte:
        search_radius=1.0mm, min_nb_points=5

    Hinweis: Parameter sind stark von der Scan-Auflösung und
    Punktdichte abhängig – Validierung mit realen Scans erforderlich.
    """

    def __init__(
        self,
        search_radius: float = 1.0,
        min_nb_points: int = 5,
        enabled: bool = True,
    ):
        """
        Args:
            search_radius: Suchradius in mm (Einheit der Punktwolke).
            min_nb_points: Mindestanzahl Nachbarn im Radius.
            enabled: Step aktiv/inaktiv.
        """
        self._search_radius = search_radius
        self._min_nb_points = min_nb_points
        self._enabled = enabled

    @property
    def name(self) -> str:
        return "radius_outlier_filter"

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _apply(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        pcd_clean, _ = pcd.remove_radius_outlier(
            nb_points=self._min_nb_points,
            radius=self._search_radius,
        )
        return pcd_clean

    def get_params(self) -> dict:
        return {
            "search_radius": self._search_radius,
            "min_nb_points": self._min_nb_points,
        }