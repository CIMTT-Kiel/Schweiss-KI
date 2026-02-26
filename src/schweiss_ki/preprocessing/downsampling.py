"""
Downsampling-Steps für die Preprocessing-Pipeline.

Enthält alle Steps, die die Punktdichte regulieren:
- VoxelGridDownsampler: Gleichmäßiges Downsampling via Voxel-Gitter
- RandomDownsampler: Zufälliges Downsampling (einfacher Fallback)
"""
from __future__ import annotations

import open3d as o3d

from .base import PreprocessingStep


class VoxelGridDownsampler(PreprocessingStep):
    """
    Voxel-Grid-Downsampling.

    Teilt den Raum in gleichmäßige Voxel der Größe voxel_size auf.
    Pro Voxel wird der Schwerpunkt aller enthaltenen Punkte behalten.

    Ergebnis: gleichmäßige Punktdichte über die gesamte Punktwolke.

    Die richtige voxel_size hängt von der Scan-Auflösung ab:
    - Zu groß: Verlust feiner Nahtgeometrie
    - Zu klein: kaum Reduktion, hohe Rechenzeiten downstream
    Empfohlener Startwert: voxel_size = 0.5mm (anpassen nach Scandaten).
    """

    def __init__(
        self,
        voxel_size: float = 0.5,
        enabled: bool = True,
    ):
        """
        Args:
            voxel_size: Kantenlänge der Voxel in mm.
            enabled: Step aktiv/inaktiv.
        """
        self._voxel_size = voxel_size
        self._enabled = enabled

    @property
    def name(self) -> str:
        return "voxel_grid_downsampler"

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _apply(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        return pcd.voxel_down_sample(voxel_size=self._voxel_size)

    def get_params(self) -> dict:
        return {"voxel_size": self._voxel_size}


class RandomDownsampler(PreprocessingStep):
    """
    Zufälliges Downsampling auf eine feste Ziel-Punktzahl.

    Einfacher Fallback für sehr große Datensätze oder wenn
    gleichmäßige Dichte keine Rolle spielt.

    Hinweis: Kein deterministisches Ergebnis ohne seed.
    """

    def __init__(
        self,
        target_points: int = 100_000,
        enabled: bool = True,
    ):
        """
        Args:
            target_points: Ziel-Punktanzahl nach Downsampling.
            enabled: Step aktiv/inaktiv.
        """
        self._target_points = target_points
        self._enabled = enabled

    @property
    def name(self) -> str:
        return "random_downsampler"

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _apply(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        n = len(pcd.points)
        if n <= self._target_points:
            return pcd
        return pcd.random_down_sample(
            sampling_ratio=self._target_points / n
        )

    def get_params(self) -> dict:
        return {"target_points": self._target_points}