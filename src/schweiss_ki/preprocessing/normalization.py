"""
Anreicherungs-Steps für die Preprocessing-Pipeline.

Enthält Steps, die Punkte mit zusätzlichen Informationen anreichern
(statt sie zu entfernen):
- NormalEstimator: Schätzt Oberflächennormalen und richtet sie aus
- Centerer: Verschiebt die Punktwolke, sodass der Schwerpunkt im Ursprung liegt
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import open3d as o3d

from .base import PreprocessingStep


class NormalEstimator(PreprocessingStep):
    """
    Schätzt Oberflächennormalen für jeden Punkt.

    Normalen werden aus der lokalen Punktnachbarschaft berechnet
    (Eigenvektor-Analyse der Kovarianzmatrix).

    Normalenorientierung:
    - "camera": Normalen zur Kameraposition ausrichten.
      Erfordert bekannte Kameraposition (scan_origin).
      Empfohlen wenn scan_origin bekannt.
    - "consistent": Normalen-Propagation für konsistente Ausrichtung.
      Fallback wenn Kameraposition unbekannt.
    - None: Keine Ausrichtung (Normalen können inkonsistent zeigen).
      Nur wenn Orientierung für den Folgeschritt irrelevant ist.

    Wichtig für:
    - RANSAC-Segmentierung (Phase 3)
    - ICP-Registrierung (AP2.2)
    """

    def __init__(
        self,
        radius: float = 2.0,
        max_nn: int = 30,
        orient_mode: Literal["camera", "consistent", None] = "consistent",
        scan_origin: Optional[list[float]] = None,
        enabled: bool = True,
    ):
        """
        Args:
            radius: Suchradius für Nachbarn in mm.
            max_nn: Maximale Nachbaranzahl (Performancegrenze).
            orient_mode: Strategie zur Normalenausrichtung.
                "camera"     – Ausrichtung zur scan_origin (Kameraposition).
                "consistent" – Propagation für konsistente Ausrichtung.
                None         – Keine Ausrichtung.
            scan_origin: Kameraposition [x, y, z] in mm.
                Nur relevant bei orient_mode="camera".
            enabled: Step aktiv/inaktiv.
        """
        self._radius = radius
        self._max_nn = max_nn
        self._orient_mode = orient_mode
        self._scan_origin = scan_origin or [0.0, 0.0, 0.0]
        self._enabled = enabled

    @property
    def name(self) -> str:
        return "normal_estimator"

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _apply(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        search_param = o3d.geometry.KDTreeSearchParamHybrid(
            radius=self._radius,
            max_nn=self._max_nn,
        )
        pcd.estimate_normals(search_param=search_param)

        if self._orient_mode == "camera":
            pcd.orient_normals_towards_camera_location(
                camera_location=np.array(self._scan_origin)
            )
        elif self._orient_mode == "consistent":
            pcd.orient_normals_consistent_tangent_plane(k=self._max_nn)

        return pcd

    def get_params(self) -> dict:
        return {
            "radius": self._radius,
            "max_nn": self._max_nn,
            "orient_mode": self._orient_mode,
            "scan_origin": self._scan_origin,
        }


class Centerer(PreprocessingStep):
    """
    Verschiebt die Punktwolke, sodass der Schwerpunkt im Ursprung liegt.

    Nützlich als Normalisierungsschritt vor Segmentierung oder
    Registrierung, um numerische Stabilität zu verbessern.

    Hinweis: Die Punktanzahl ändert sich nicht – es werden keine
    Punkte entfernt, nur verschoben.
    """

    def __init__(self, enabled: bool = True):
        """
        Args:
            enabled: Step aktiv/inaktiv.
        """
        self._enabled = enabled

    @property
    def name(self) -> str:
        return "centerer"

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _apply(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        centroid = pcd.get_center()
        pcd_centered = pcd.translate(-centroid)
        return pcd_centered

    def get_params(self) -> dict:
        return {}