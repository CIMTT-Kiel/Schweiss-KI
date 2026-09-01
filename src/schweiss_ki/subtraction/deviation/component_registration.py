"""
ComponentRegistration – werkstückweise Registrierung für relative Lage-Features.

Idee (Phase 3 AP2.3):
    Statt das gesamte Bauteil auf einmal zu registrieren, wird jedes
    Werkstück einzeln gegen sein CAD-Pendant ausgerichtet. Die Differenz
    der beiden resultierenden Transformationen beschreibt, wie die
    Werkstücke im realen Bauteil gegeneinander positioniert sind:
    drei Translations- und drei Rotationsfreiheitsgrade.

    Diese sechs Größen sind invariant gegen die Lage des Bauteils in
    der Scanner-Kammer – sie beschreiben ausschließlich die relative
    Fertigungsabweichung zwischen den Werkstücken.

Voraussetzung:
    Scan liegt bereits im CAD-Koordinatensystem (Gesamt-Registrierung
    ist Vor-Bedingung dieses Steps). Die Trennung der Werkstücke
    erfolgt aktuell räumlich anhand einer konfigurierbaren Achse und
    Position (Default: Y = 0 im CAD-Koordinatensystem).

Registrierung pro Werkstück:
    Nur ICPFine. Ein separates CoarsePCA wäre schädlich, weil ein
    einzelnes Werkstück eine geometrisch weitgehend uniforme Platte
    ist: PCA hat dort mehrere gültige Achsen-Ausrichtungen (typisch
    180°-Ambiguität), was Registrierung ins Chaos schickt. Die
    Ausgangslage ist ohnehin schon nahe am Ziel – ICP mit enger
    max_correspondence_distance reicht.

Ergebnis wird in DeviationData.component_registration abgelegt.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import open3d as o3d

from ..base import DeviationStep
from ..registration.pipeline import RegistrationPipeline
from ..reports import DeviationData

logger = logging.getLogger(__name__)


class ComponentRegistration(DeviationStep):
    """Werkstückweise Registrierung mit Ableitung der relativen Lage.

    Args:
        split_axis: Achse für die räumliche Trennung der Werkstücke.
            0 = X, 1 = Y, 2 = Z. Default 1 (Y).
        split_value: Trennposition entlang der Achse. Default 0.0.
        component_a_side: "positive" (A hat Achsen-Werte > split_value)
            oder "negative". Default "positive".
        icp_max_correspondence_distance: mm. Klein wählen (0.3–1.0), da
            Ausgangslage nahe am Ziel. Default 0.5.
        icp_max_iteration: Iterationslimit für ICPFine. Default 50.
        icp_anchor_labels: Punkt-Labels für Anker-Beschränkung in ICP.
            Default (0, 1, 2) – Werkstück-Oberseite und Flanke A/B.
        enabled: Step-Aktivierung.
    """

    def __init__(
        self,
        split_axis: int = 1,
        split_value: float = 0.0,
        component_a_side: str = "positive",
        icp_max_correspondence_distance: float = 0.5,
        icp_max_iteration: int = 50,
        icp_anchor_labels: Optional[List[int]] = None,
        enabled: bool = True,
    ):
        self._enabled = enabled
        self.split_axis = int(split_axis)
        self.split_value = float(split_value)
        if component_a_side not in ("positive", "negative"):
            raise ValueError(
                f"component_a_side muss 'positive' oder 'negative' sein, "
                f"war '{component_a_side}'."
            )
        self.component_a_side = component_a_side
        self.icp_max_correspondence_distance = float(icp_max_correspondence_distance)
        self.icp_max_iteration = int(icp_max_iteration)
        self.icp_anchor_labels = (
            tuple(icp_anchor_labels) if icp_anchor_labels is not None else (0, 1, 2)
        )

        if self.split_axis not in (0, 1, 2):
            raise ValueError(f"split_axis muss 0, 1 oder 2 sein, war {self.split_axis}.")

    @property
    def name(self) -> str:
        return "component_registration"

    def get_params(self) -> Dict[str, Any]:
        return {
            "split_axis": self.split_axis,
            "split_value": self.split_value,
            "component_a_side": self.component_a_side,
            "icp_max_correspondence_distance": self.icp_max_correspondence_distance,
            "icp_max_iteration": self.icp_max_iteration,
            "icp_anchor_labels": list(self.icp_anchor_labels),
        }

    # ── Hauptlogik ────────────────────────────────────────────────────

    def _apply(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        data: DeviationData,
        source_labels: Optional[np.ndarray] = None,
        target_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        source_pts = np.asarray(source.points)
        target_pts = np.asarray(target.points)

        # Räumliche Trennung entlang split_axis
        if self.component_a_side == "positive":
            src_mask_a = source_pts[:, self.split_axis] >= self.split_value
            tgt_mask_a = target_pts[:, self.split_axis] >= self.split_value
        else:
            src_mask_a = source_pts[:, self.split_axis] < self.split_value
            tgt_mask_a = target_pts[:, self.split_axis] < self.split_value

        src_mask_b = ~src_mask_a
        tgt_mask_b = ~tgt_mask_a

        n_src_a, n_src_b = int(src_mask_a.sum()), int(src_mask_b.sum())
        n_tgt_a, n_tgt_b = int(tgt_mask_a.sum()), int(tgt_mask_b.sum())

        if n_src_a < 100 or n_src_b < 100 or n_tgt_a < 100 or n_tgt_b < 100:
            logger.warning(
                f"  ComponentRegistration: zu wenig Punkte pro Werkstück "
                f"(A: src={n_src_a}, tgt={n_tgt_a} | B: src={n_src_b}, tgt={n_tgt_b}). "
                f"Step übersprungen."
            )
            return {"skipped": True}

        # Werkstück-Wolken extrahieren
        source_a = source.select_by_index(np.where(src_mask_a)[0].tolist())
        source_b = source.select_by_index(np.where(src_mask_b)[0].tolist())
        target_a = target.select_by_index(np.where(tgt_mask_a)[0].tolist())
        target_b = target.select_by_index(np.where(tgt_mask_b)[0].tolist())

        # Labels für Anker-Beschränkung entsprechend filtern (falls vorhanden)
        src_labels_a = source_labels[src_mask_a] if source_labels is not None else None
        src_labels_b = source_labels[src_mask_b] if source_labels is not None else None

        # Pro Werkstück nur ICPFine (siehe Docstring oben)
        pipeline_a = self._build_pipeline()
        pipeline_b = self._build_pipeline()

        logger.info(
            f"  ComponentRegistration: Werkstück A ({n_src_a:,} src / "
            f"{n_tgt_a:,} tgt) und B ({n_src_b:,} src / {n_tgt_b:,} tgt)"
        )

        _, report_a = pipeline_a.run(source_a, target_a, source_labels=src_labels_a)
        _, report_b = pipeline_b.run(source_b, target_b, source_labels=src_labels_b)

        T_a = report_a.final_transform
        T_b = report_b.final_transform

        # Relative Transformation: T_rel = T_b · T_a⁻¹
        T_rel = T_b @ np.linalg.inv(T_a)

        # Rotationsanteil und roher Translationsanteil aus T_rel
        R_rel = T_rel[:3, :3]
        t_raw = T_rel[:3, 3]
        rotation_deg = self._rotation_matrix_to_euler_xyz_deg(R_rel)

        # Zerlegung der Translation um den Werkstück-B-Schwerpunkt:
        #
        # Die T_rel[:3, 3]-Komponente enthält bei Rotationen um den Ursprung
        # zusätzlich einen Anteil, der durch die Verschiebung des Werkstück-
        # Schwerpunkts zustande kommt (Off-Center-Rotation). Für eine
        # sauber interpretierbare Zerlegung wird dieser Anteil abgezogen:
        #
        #   t_pure = t_raw - (I - R_rel) · c_B
        #
        # wobei c_B der Schwerpunkt des Werkstück-B-Targets ist.
        # Damit gilt: bei reiner Rotation ist t_pure ≈ 0, bei reiner
        # Translation ist t_pure = t_raw.
        target_b_center = np.mean(np.asarray(target_b.points), axis=0)
        t_pure = t_raw - (np.eye(3) - R_rel) @ target_b_center

        result = {
            "translation_mm": {
                "x": float(t_pure[0]),
                "y": float(t_pure[1]),
                "z": float(t_pure[2]),
            },
            "translation_raw_mm": {
                "x": float(t_raw[0]),
                "y": float(t_raw[1]),
                "z": float(t_raw[2]),
            },
            "rotation_center_mm": {
                "x": float(target_b_center[0]),
                "y": float(target_b_center[1]),
                "z": float(target_b_center[2]),
            },
            "rotation_deg": {
                "x": float(rotation_deg[0]),
                "y": float(rotation_deg[1]),
                "z": float(rotation_deg[2]),
            },
            "T_a": T_a.tolist(),
            "T_b": T_b.tolist(),
            "T_relative": T_rel.tolist(),
            "residual_a_mm": report_a.final_residual,
            "residual_b_mm": report_b.final_residual,
            "n_points_a": {"source": n_src_a, "target": n_tgt_a},
            "n_points_b": {"source": n_src_b, "target": n_tgt_b},
        }

        data.component_registration = result

        logger.info(
            f"    Residuum: A={report_a.final_residual:.3f}mm, "
            f"B={report_b.final_residual:.3f}mm"
        )
        logger.info(
            f"    Relative Lage (Rotation um Werkstück-B-Schwerpunkt): "
            f"Δx={t_pure[0]:+.3f}mm, Δy={t_pure[1]:+.3f}mm, "
            f"Δz={t_pure[2]:+.3f}mm | "
            f"rot_x={rotation_deg[0]:+.3f}°, rot_y={rotation_deg[1]:+.3f}°, "
            f"rot_z={rotation_deg[2]:+.3f}°"
        )

        return {
            "translation_mm": result["translation_mm"],
            "translation_raw_mm": result["translation_raw_mm"],
            "rotation_deg": result["rotation_deg"],
            "residual_a_mm": report_a.final_residual,
            "residual_b_mm": report_b.final_residual,
        }

    # ── Helpers ───────────────────────────────────────────────────────

    def _build_pipeline(self) -> RegistrationPipeline:
        """Baut die Registrierungs-Pipeline für ein einzelnes Werkstück.

        XEdgeAlign vor ICPFine – kein CoarsePCA (PCA ist an einer einzelnen
        Platte instabil). XEdgeAlign richtet die Naht-Längsrichtung (X) über
        die Kanten aus, bevor ICP verfeinert: Point-to-Plane-ICP ist in X
        schwach (prismatische Nut, X-Signal nur an den Enden) und bleibt sonst
        mit einem Restversatz stehen. Da jedes Werkstück einzeln gegen das CAD
        ausgerichtet wird, ergibt die Differenz der beiden Rest-Versätze einen
        systematischen Fehler in der gemessenen Relativlage (früher ~0.2 mm in
        Tx). XEdgeAlign nimmt ICP diesen X-Sprung ab.
        """
        from ..registration import ICPFine, XEdgeAlign
        return RegistrationPipeline([
            XEdgeAlign(anchor_labels=list(self.icp_anchor_labels)),
            ICPFine(
                max_correspondence_distance=self.icp_max_correspondence_distance,
                max_iteration=self.icp_max_iteration,
                anchor_labels=list(self.icp_anchor_labels),
            ),
        ])

    @staticmethod
    def _rotation_matrix_to_euler_xyz_deg(R: np.ndarray) -> np.ndarray:
        """Rotationsmatrix → Euler-Winkel (XYZ-Konvention, in Grad).

        Konvention: R = R_x(α) · R_y(β) · R_z(γ)

        Für V-Naht-Bauteile bleiben die Winkel klein (< 5°), Gimbal Lock
        ist nicht praxisrelevant.
        """
        sy = -R[2, 0]
        cy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

        if cy > 1e-6:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(sy, cy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(sy, cy)
            z = 0.0

        return np.rad2deg(np.array([x, y, z]))