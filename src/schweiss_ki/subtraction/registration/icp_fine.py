"""
ICPFine – Fein-Registrierung via Point-to-Plane ICP (Open3D).

Setzt CoarsePCA voraus: erwartet, dass die Restrotation bereits in der
typischen ICP-Konvergenzregion liegt (< 5°) und der Translationsrest
in der Größenordnung der max_correspondence_distance.

Point-to-Plane wurde gewählt, weil:
    - Das CAD-Target Normalen aus der API mitliefert.
    - V-Naht-Geometrie ist überwiegend planar (Flanken + Oberseite) –
      Point-to-Plane konvergiert auf planaren Flächen deutlich schneller
      und genauer als Point-to-Point.

Anker-Beschränkung der Source (Labels {0, 1, 2}): die Spalt-Region und
Sub-Gap-Artefakte haben keine korrespondierenden CAD-Punkte und würden
die Korrespondenzsuche in die Irre führen.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d

from ..base import RegistrationStep

logger = logging.getLogger(__name__)


class ICPFine(RegistrationStep):
    """Point-to-Plane ICP als Fein-Registrierung.

    Args:
        max_correspondence_distance: Maximaler Abstand für gültige
            Korrespondenzen (mm). Nach CoarsePCA typischerweise ~1.0 mm
            geeignet; engerer Wert verschärft die Optimierung, kann aber
            Konvergenz verhindern wenn die Init zu schlecht ist.
        max_iteration: Maximale ICP-Iterationen. Default 50.
        relative_fitness: Konvergenzkriterium – Stop wenn relative
            Fitness-Änderung darunter fällt.
        relative_rmse: Konvergenzkriterium – Stop wenn relative
            RMSE-Änderung darunter fällt.
        anchor_labels: Wenn gegeben, wird ICP nur auf Source-Punkten mit
            diesen Labels berechnet. Default (0, 1, 2) – Hintergrund + Flanken,
            Spalt-Region (3) und Sub-Gap-Artefakte (4) werden ausgeschlossen.
        evaluation_samples: Subsample-Größe für den Residual-Report.
        enabled: Step-Aktivierung.
        random_seed: Seed für Subsampling-Reproduzierbarkeit.
    """

    def __init__(
        self,
        max_correspondence_distance: float = 1.0,
        max_iteration: int = 50,
        relative_fitness: float = 1e-6,
        relative_rmse: float = 1e-6,
        anchor_labels: Optional[Sequence[int]] = (0, 1, 2),
        evaluation_samples: int = 5_000,
        enabled: bool = True,
        random_seed: int = 0,
    ):
        self._enabled = enabled
        self.max_correspondence_distance = float(max_correspondence_distance)
        self.max_iteration = int(max_iteration)
        self.relative_fitness = float(relative_fitness)
        self.relative_rmse = float(relative_rmse)
        self.anchor_labels = (
            None if anchor_labels is None else tuple(anchor_labels)
        )
        self.evaluation_samples = int(evaluation_samples)
        self.random_seed = int(random_seed)

    @property
    def name(self) -> str:
        return "icp_fine"

    def get_params(self) -> Dict[str, Any]:
        return {
            "max_correspondence_distance": self.max_correspondence_distance,
            "max_iteration": self.max_iteration,
            "relative_fitness": self.relative_fitness,
            "relative_rmse": self.relative_rmse,
            "anchor_labels": self.anchor_labels,
        }

    # ── Hauptlogik ────────────────────────────────────────────────────

    def _apply(
        self,
        source_aligned: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        source_labels: Optional[np.ndarray] = None,
        target_labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not target.has_normals():
            raise ValueError(
                "ICPFine (point-to-plane) benötigt Normalen am Target. "
                "CAD-Wolken aus der API liefern diese mit. Bei selbst "
                "erzeugten Targets vor ICP estimate_normals() aufrufen."
            )

        # Source auf Anker-Punkte beschränken
        if self.anchor_labels is not None and source_labels is not None:
            mask = np.isin(source_labels, self.anchor_labels)
            idx = np.where(mask)[0]
            source_subset = source_aligned.select_by_index(idx.tolist())
        else:
            source_subset = source_aligned

        if len(source_subset.points) < 10:
            raise ValueError(
                f"Zu wenig Source-Anker-Punkte für ICP "
                f"(n={len(source_subset.points)})."
            )

        result = o3d.pipelines.registration.registration_icp(
            source=source_subset,
            target=target,
            max_correspondence_distance=self.max_correspondence_distance,
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.max_iteration,
                relative_fitness=self.relative_fitness,
                relative_rmse=self.relative_rmse,
            ),
        )

        delta_transform = np.asarray(result.transformation)

        artifacts = {
            "fitness": float(result.fitness),
            "inlier_rmse": float(result.inlier_rmse),
            "max_correspondence_distance": self.max_correspondence_distance,
            "anchor_count_source": int(len(source_subset.points)),
            "target_count": int(len(target.points)),
        }

        logger.debug(
            f"  ICPFine: fitness={result.fitness:.3f}, "
            f"inlier_rmse={result.inlier_rmse:.3f} mm, "
            f"source_anchors={len(source_subset.points):,}"
        )

        return delta_transform, artifacts

    def compute_residual(
        self,
        source_after: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        source_labels: Optional[np.ndarray] = None,
        target_labels: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        """Mittlerer NN-Abstand source → target nach Anwendung der Transform."""
        all_src = np.asarray(source_after.points)
        n_eval = min(self.evaluation_samples, len(all_src))
        rng = np.random.default_rng(self.random_seed)
        idx = rng.choice(len(all_src), n_eval, replace=False)
        kdtree = o3d.geometry.KDTreeFlann(target)

        moved = all_src[idx]
        sq_dists = np.empty(len(moved))
        for i, p in enumerate(moved):
            _, _, d2 = kdtree.search_knn_vector_3d(p, 1)
            sq_dists[i] = d2[0]
        return float(np.sqrt(sq_dists).mean())