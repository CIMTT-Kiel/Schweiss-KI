"""
GapProfile – Wurzelspalt-Profil entlang der Naht-Längsrichtung.

Methodik (an der Deckflächen-Ebene verankert):
    - Je Werkstück wird die Deckflächen-Ebene per RANSAC gefittet. Die Ebene
      des Referenz-Werkstücks definiert die Tiefe d = 0.
    - Für jede der beiden Flanken werden die segmentierten Punkte entlang der
      Naht-Längsachse in Bins zerlegt.
    - Pro Bin und Flanke wird q(d) = q₀ + m·d gefittet, mit q = Position auf
      der gap_axis und d = Tiefe unter der Referenzebene. Getrennte Fits, damit
      eine fehlende Gegenflanke den anderen Fit nicht beeinträchtigt.
    - Die Spaltbreite ist die Differenz beider Geraden, ausgewertet an der
      tiefsten beidseitig besetzten Tiefe – ein Messwert, keine Extrapolation.

Warum nicht mehr auf z = 0 extrapoliert wird:
    Die frühere Methode zielte auf vertical_axis = 0 im Koordinatensystem der
    registrierten Wolke. Damit ging jeder Registrierungs-Höhenversatz dz mit
    dw/dz = 2·tan(α) in die Spaltbreite ein – bei der 90°-Naht verdoppelt.
    Auf den 61 synthetischen Fällen war das die alleinige Ursache der
    verbliebenen Untererfassung (Korrelation 0.99998 zwischen Fehler und
    -2·dz), mit Fehlern bis 1.3 mm. Die Verankerung an der Deckfläche lässt dz
    strukturell herausfallen: der Registrierungseinfluss sank auf max. 0.008 mm
    über alle 61 Fälle. Details in docs/fehleranalyse_achsen_und_registrierung.md.

Voraussetzungen:
    - Scan ist bereits ausgerichtet (XEdgeAlign + ICPFine vorher in Pipeline).
    - source_labels enthält Flanken- UND Background-Labels (aus AP2.1) –
      letztere für die Deckflächen-Ebenen.
    - Achsen-Konvention: Naht entlang einer konfigurierbaren Achse,
      Spalt entlang einer dazu senkrechten Achse, Vertikale = Z.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import open3d as o3d

from ..base import DeviationStep
from ..reports import DeviationData

logger = logging.getLogger(__name__)


def _plane_to_dict(plane: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Ebenen-Dict JSON-tauglich machen (numpy-Normale → Liste)."""
    if plane is None:
        return None
    out = dict(plane)
    out["normal"] = [float(x) for x in plane["normal"]]
    return out


class GapProfile(DeviationStep):
    """Wurzelspalt-Profil entlang der Naht-Längsrichtung."""

    def __init__(
        self,
        flank_a_label: int = 1,
        flank_b_label: int = 2,
        background_label: int = 0,
        seam_axis: int = 0,
        gap_axis: int = 1,
        vertical_axis: int = 2,
        n_bins: int = 20,
        edge_margin: float = 10.0,
        min_points_per_bin: int = 3,
        # ── Deckflächen-Verankerung ───────────────────────────────────
        reference_side: str = "positive",
        top_plane_ransac_threshold: float = 0.25,
        top_plane_max_iterations: int = 1000,
        top_plane_min_inlier_ratio: float = 0.5,
        # ── Flankenfits über Tiefenband ───────────────────────────────
        flank_depth_min: float = 0.5,
        flank_depth_max_quantile: float = 0.95,
        min_points_per_flank_bin: int = 10,
        min_depth_span: float = 1.0,
        enabled: bool = True,
    ):
        self._enabled = enabled
        self.flank_a_label = int(flank_a_label)
        self.flank_b_label = int(flank_b_label)
        self.background_label = int(background_label)
        self.seam_axis = int(seam_axis)
        self.gap_axis = int(gap_axis)
        self.vertical_axis = int(vertical_axis)
        self.n_bins = int(n_bins)
        self.edge_margin = float(edge_margin)
        self.min_points_per_bin = int(min_points_per_bin)

        if reference_side not in ("positive", "negative"):
            raise ValueError(
                f"reference_side muss 'positive' oder 'negative' sein, ist: {reference_side}"
            )
        self.reference_side = reference_side
        self.top_plane_ransac_threshold = float(top_plane_ransac_threshold)
        self.top_plane_max_iterations = int(top_plane_max_iterations)
        self.top_plane_min_inlier_ratio = float(top_plane_min_inlier_ratio)

        self.flank_depth_min = float(flank_depth_min)
        self.flank_depth_max_quantile = float(flank_depth_max_quantile)
        self.min_points_per_flank_bin = int(min_points_per_flank_bin)
        self.min_depth_span = float(min_depth_span)

        if len({self.seam_axis, self.gap_axis, self.vertical_axis}) != 3:
            raise ValueError(
                "seam_axis, gap_axis, vertical_axis müssen drei verschiedene "
                f"Achsen sein, sind aber: {self.seam_axis}, {self.gap_axis}, "
                f"{self.vertical_axis}"
            )

    @property
    def name(self) -> str:
        return "gap_profile"

    def get_params(self) -> Dict[str, Any]:
        return {
            "flank_a_label": self.flank_a_label,
            "flank_b_label": self.flank_b_label,
            "background_label": self.background_label,
            "seam_axis": self.seam_axis,
            "gap_axis": self.gap_axis,
            "vertical_axis": self.vertical_axis,
            "n_bins": self.n_bins,
            "edge_margin": self.edge_margin,
            "min_points_per_bin": self.min_points_per_bin,
            "reference_side": self.reference_side,
            "top_plane_ransac_threshold": self.top_plane_ransac_threshold,
            "top_plane_max_iterations": self.top_plane_max_iterations,
            "top_plane_min_inlier_ratio": self.top_plane_min_inlier_ratio,
            "flank_depth_min": self.flank_depth_min,
            "flank_depth_max_quantile": self.flank_depth_max_quantile,
            "min_points_per_flank_bin": self.min_points_per_flank_bin,
            "min_depth_span": self.min_depth_span,
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
        if source_labels is None:
            raise ValueError(
                "GapProfile benötigt source_labels (Flanken-Segmentierung). "
                "AP2.1-Segmentierung muss vorher gelaufen sein."
            )

        pts = np.asarray(source.points)
        flank_a = pts[source_labels == self.flank_a_label]
        flank_b = pts[source_labels == self.flank_b_label]

        if len(flank_a) == 0 or len(flank_b) == 0:
            logger.warning(
                f"  GapProfile: leere Flanken (A={len(flank_a)}, B={len(flank_b)}). "
                f"Step übersprungen."
            )
            return {"n_bins_valid": 0, "skipped": True}

        # Bin-Grenzen entlang der Naht-Achse, Margin gegen Heftpunkte
        seam_min = max(flank_a[:, self.seam_axis].min(),
                       flank_b[:, self.seam_axis].min()) + self.edge_margin
        seam_max = min(flank_a[:, self.seam_axis].max(),
                       flank_b[:, self.seam_axis].max()) - self.edge_margin

        if seam_max <= seam_min:
            logger.warning(
                f"  GapProfile: edge_margin ({self.edge_margin} mm) zu groß für "
                f"verbleibende Naht-Länge. Step übersprungen."
            )
            return {"n_bins_valid": 0, "skipped": True}

        bin_edges = np.linspace(seam_min, seam_max, self.n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # ── Deckflächen-Ebenen je Werkstück ───────────────────────────
        # Die A/B-Trennung kommt geometrisch (Vorzeichen auf gap_axis), nicht
        # aus den Labels: 'background' ist EIN Label für beide Werkstücke, und
        # background_remover läuft als erster Step, wenn noch nichts anderes
        # klassifiziert ist.
        bg = pts[source_labels == self.background_label]
        pos = bg[bg[:, self.gap_axis] >= 0.0]
        neg = bg[bg[:, self.gap_axis] < 0.0]
        ref_pts, opp_pts = (pos, neg) if self.reference_side == "positive" else (neg, pos)

        ref_plane = self._fit_top_plane(ref_pts)
        opp_plane = self._fit_top_plane(opp_pts)

        anchored = ref_plane is not None
        gate_msg = None
        if anchored and ref_plane["inlier_ratio"] < self.top_plane_min_inlier_ratio:
            gate_msg = (
                f"inlier_ratio {ref_plane['inlier_ratio']:.2f} < "
                f"{self.top_plane_min_inlier_ratio}"
            )
            anchored = False
        if ref_plane is None:
            gate_msg = f"zu wenige Background-Punkte auf der Referenzseite ({len(ref_pts)})"

        if not anchored:
            logger.warning(
                f"  GapProfile: Deckflächen-Verankerung nicht möglich ({gate_msg}). "
                f"Tiefenbezogene Auswertung wird übersprungen."
            )

        # ── Flankenfits über Tiefe unter der Referenz-Deckfläche ──────
        n_valid_depth = 0
        prof = {s: {k: np.full(self.n_bins, np.nan) for k in
                    ("q0", "slope", "r2", "rms", "n_points", "d_lo", "d_hi")}
                for s in ("flank_a", "flank_b")}
        gap_root = np.full(self.n_bins, np.nan)
        d_root = np.full(self.n_bins, np.nan)

        if anchored:
            depth_a = self._depth_below(ref_plane, flank_a)
            depth_b = self._depth_below(ref_plane, flank_b)
            # Untergrenze datengetrieben je Flanke (Scan-Dichte, nicht fest)
            d_max_a = float(np.quantile(depth_a, self.flank_depth_max_quantile))
            d_max_b = float(np.quantile(depth_b, self.flank_depth_max_quantile))

            for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
                ma = (flank_a[:, self.seam_axis] >= lo) & (flank_a[:, self.seam_axis] < hi)
                mb = (flank_b[:, self.seam_axis] >= lo) & (flank_b[:, self.seam_axis] < hi)
                fa = self._fit_flank_in_bin(flank_a[ma], depth_a[ma], d_max_a)
                fb = self._fit_flank_in_bin(flank_b[mb], depth_b[mb], d_max_b)
                for side, fit in (("flank_a", fa), ("flank_b", fb)):
                    if fit is not None:
                        for k, v in fit.items():
                            prof[side][k][i] = v
                if fa is None or fb is None:
                    continue
                # Wurzelspalt als MESSWERT an der tiefsten beidseitig besetzten
                # Tiefe – die Geraden werden dort ausgewertet, nicht extrapoliert.
                dr = min(fa["d_hi"], fb["d_hi"])
                d_root[i] = dr
                gap_root[i] = (fb["q0"] + fb["slope"] * dr) - (fa["q0"] + fa["slope"] * dr)
                n_valid_depth += 1

        if anchored:
            logger.info(
                f"  GapProfile: verankert an Deckfläche ({self.reference_side}), "
                f"inlier {ref_plane['inlier_ratio']:.2f}, rms {ref_plane['rms_mm']:.3f} mm; "
                f"{n_valid_depth}/{self.n_bins} Bins mit Wurzelspalt"
            )

        # In DeviationData schreiben
        data.gap_profile = {
            "seam_axis_centers": bin_centers,
            "seam_axis": self.seam_axis,
            "gap_axis": self.gap_axis,
            "vertical_axis": self.vertical_axis,
            "anchored": anchored,
            "flank_a_profile": prof["flank_a"],
            "flank_b_profile": prof["flank_b"],
            "gap_root_widths": gap_root,
            "d_root": d_root,
            "reference_plane": _plane_to_dict(ref_plane),
            "opposite_plane": _plane_to_dict(opp_plane),
            "opposite_vs_reference": (
                self._relative_pose(ref_plane, opp_plane, opp_pts)
                if (ref_plane is not None and opp_plane is not None and len(opp_pts))
                else None
            ),
        }

        artifacts: Dict[str, Any] = {
            "n_bins_total": self.n_bins,
            "n_bins_valid_depth": n_valid_depth,
            "anchored": anchored,
            "seam_range_used": (float(seam_min), float(seam_max)),
        }
        if ref_plane is not None:
            artifacts["reference_plane_inlier_ratio"] = ref_plane["inlier_ratio"]
            artifacts["reference_plane_rms_mm"] = ref_plane["rms_mm"]
        if opp_plane is not None:
            artifacts["opposite_plane_inlier_ratio"] = opp_plane["inlier_ratio"]
            artifacts["opposite_plane_rms_mm"] = opp_plane["rms_mm"]
        rel = data.gap_profile["opposite_vs_reference"]
        if rel is not None:
            artifacts["kantenversatz_mm"] = rel["height_offset_mm"]
            artifacts["relative_tilt_deg"] = rel["tilt_total_deg"]

        if n_valid_depth > 0:
            artifacts.update({
                "gap_root_min_mm": float(np.nanmin(gap_root)),
                "gap_root_max_mm": float(np.nanmax(gap_root)),
                "gap_root_mean_mm": float(np.nanmean(gap_root)),
                "gap_root_std_mm": float(np.nanstd(gap_root)),
            })
            # Flankenwinkel als Qualitätsmerkmal – NICHT als bekannt annehmen.
            # slope = dq/dd; Winkel zur Vertikalen = atan(|slope|).
            ang = {}
            for side in ("flank_a", "flank_b"):
                sl = prof[side]["slope"]
                if np.isfinite(sl).any():
                    ang[side] = float(np.degrees(np.arctan(np.abs(np.nanmean(sl)))))
                    artifacts[f"{side}_angle_deg"] = ang[side]
                    artifacts[f"{side}_r2_min"] = float(np.nanmin(prof[side]["r2"]))
                    artifacts[f"{side}_rms_max_mm"] = float(np.nanmax(prof[side]["rms"]))
            if len(ang) == 2:
                artifacts["flank_angle_asymmetry_deg"] = abs(ang["flank_a"] - ang["flank_b"])

        return artifacts

    # ── Deckflächen-Verankerung ───────────────────────────────────────

    def _fit_top_plane(self, pts: np.ndarray) -> Optional[Dict[str, Any]]:
        """RANSAC-Ebenenfit auf eine Deckflächen-Punktmenge.

        RANSAC statt Least-Squares, weil bei realen Scans Spritzer und
        Reflexionen genau auf der Deckfläche sitzen und ein LSQ-Fit dort
        mitwandert. Der Fit-Fehler geht mit 2·tan(α) in die Spaltbreite ein
        (siehe Klassen-Docstring) – deshalb werden inlier_ratio und rms als
        Gütemaße mitgegeben.

        Returns:
            dict mit normal (nach oben orientiert), d, inlier_ratio, rms,
            n_points. None, wenn zu wenige Punkte.
        """
        if len(pts) < 3:
            return None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        model, inliers = pcd.segment_plane(
            distance_threshold=self.top_plane_ransac_threshold,
            ransac_n=3,
            num_iterations=self.top_plane_max_iterations,
        )
        n = np.asarray(model[:3], dtype=float)
        d = float(model[3])
        norm = np.linalg.norm(n)
        if norm == 0.0:
            return None
        n, d = n / norm, d / norm

        # Normale konsistent nach oben orientieren
        if n[self.vertical_axis] < 0:
            n, d = -n, -d

        inl = np.asarray(inliers, dtype=int)
        rms = float(np.sqrt(np.mean((pts[inl] @ n + d) ** 2))) if len(inl) else float("nan")
        return {
            "normal": n,
            "d": d,
            "inlier_ratio": float(len(inl) / len(pts)),
            "rms_mm": rms,
            "n_points": int(len(pts)),
        }

    def _depth_below(self, plane: Dict[str, Any], pts: np.ndarray) -> np.ndarray:
        """Tiefe unter der Referenzebene, positiv nach unten."""
        return -(pts @ plane["normal"] + plane["d"])

    def _relative_pose(
        self, ref: Dict[str, Any], opp: Dict[str, Any], opp_pts: np.ndarray
    ) -> Dict[str, float]:
        """Lage der Gegenseite relativ zur Referenz-Deckfläche.

        Bildet Kantenversatz ab (Höhenversatz) und relative Verkippung,
        zerlegt in Naht-Längs- und Querrichtung. Eigenes Qualitätsmerkmal –
        geht bewusst NICHT in die Höhenreferenz ein, damit ein verkipptes
        Gegenstück die Spaltmessung nicht verfälscht.
        """
        offset = float(np.mean(self._depth_below(ref, opp_pts)))
        n_r, n_o = ref["normal"], opp["normal"]
        cos = float(np.clip(n_r @ n_o, -1.0, 1.0))
        # Kippkomponenten: Projektion der Normalen-Differenz auf die Achsen
        dn = n_o - n_r
        v = self.vertical_axis
        return {
            "height_offset_mm": -offset,   # positiv = Gegenseite liegt höher
            "tilt_total_deg": float(np.degrees(np.arccos(cos))),
            "tilt_along_seam_deg": float(
                np.degrees(np.arctan2(dn[self.seam_axis], n_r[v]))
            ),
            "tilt_across_gap_deg": float(
                np.degrees(np.arctan2(dn[self.gap_axis], n_r[v]))
            ),
        }

    # ── Flankenfits über Tiefe ────────────────────────────────────────

    @staticmethod
    def _fit_line(d: np.ndarray, q: np.ndarray) -> Optional[Dict[str, float]]:
        """Geradenfit q(d) = q0 + m·d mit Gütemaß.

        Returns None bei zu wenig Spreizung in d (Fit wäre unbestimmt).
        """
        if len(d) < 2 or float(d.max() - d.min()) <= 0:
            return None
        try:
            m, q0 = np.polyfit(d, q, 1)
        except (np.linalg.LinAlgError, ValueError):
            return None
        resid = q - (q0 + m * d)
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((q - q.mean()) ** 2))
        return {
            "q0": float(q0),
            "slope": float(m),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "rms": float(np.sqrt(ss_res / len(d))),
            "n_points": int(len(d)),
            "d_lo": float(d.min()),
            "d_hi": float(d.max()),
        }

    def _fit_flank_in_bin(
        self, pts: np.ndarray, depth: np.ndarray, d_max: float
    ) -> Optional[Dict[str, float]]:
        """Flankengerade eines Bins über dem zulässigen Tiefenband."""
        band = (depth >= self.flank_depth_min) & (depth <= d_max)
        if int(band.sum()) < self.min_points_per_flank_bin:
            return None
        d = depth[band]
        if float(d.max() - d.min()) < self.min_depth_span:
            return None
        return self._fit_line(d, pts[band, self.gap_axis])
