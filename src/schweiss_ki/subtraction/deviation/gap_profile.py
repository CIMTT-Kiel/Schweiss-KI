"""
GapProfile – Wurzelspalt-Profil entlang der Naht-Längsrichtung.

Methodik:
    - Für jede der beiden Flanken (Labels A und B) werden die segmentierten
      Punkte entlang der Naht-Längsachse in Bins zerlegt.
    - Pro Bin und Flanke wird eine Geradenanpassung y(z) = m·z + y₀ durchgeführt.
    - Der Y-Schnittpunkt der Geraden mit z = 0 liefert den virtuellen
      Wurzelpunkt der Flanke (die echten Wurzelpunkte fehlen typischerweise,
      weil der CMM-Strahl dort keine Messung mehr aufnimmt).
    - Die Spaltbreite pro Bin ist der Y-Abstand der beiden Wurzelpunkte.

Voraussetzungen:
    - Scan ist bereits ausgerichtet (CoarsePCA + ICPFine vorher in Pipeline).
    - source_labels enthält die Flanken-Labels (typisch 1 und 2 aus AP2.1).
    - Achsen-Konvention: Naht entlang einer konfigurierbaren Achse,
      Spalt entlang einer dazu senkrechten Achse, Vertikale = Z.

Konfigurierbar:
    flank_a_label, flank_b_label     – welche Labels die Flanken bezeichnen
    seam_axis, gap_axis, vertical_axis – Achsen-Konvention (0=X, 1=Y, 2=Z)
    n_bins                             – Anzahl Bins entlang der Naht
    edge_margin                        – mm an beiden Enden ausklammern (Heftpunkte)
    min_points_per_bin                 – Mindest-Punktzahl pro Flanke und Bin
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
        emit_legacy_gap_widths: bool = True,
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
        self.emit_legacy_gap_widths = bool(emit_legacy_gap_widths)

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
            "emit_legacy_gap_widths": self.emit_legacy_gap_widths,
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

        gap_widths = np.full(self.n_bins, np.nan)
        y_a0 = np.full(self.n_bins, np.nan)
        y_b0 = np.full(self.n_bins, np.nan)

        for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            mask_a = (flank_a[:, self.seam_axis] >= lo) & (flank_a[:, self.seam_axis] < hi)
            mask_b = (flank_b[:, self.seam_axis] >= lo) & (flank_b[:, self.seam_axis] < hi)
            sub_a = flank_a[mask_a]
            sub_b = flank_b[mask_b]

            ya = self._extrapolate_to_z0(sub_a)
            yb = self._extrapolate_to_z0(sub_b)

            if not (np.isnan(ya) or np.isnan(yb)):
                gap_widths[i] = abs(ya - yb)
                y_a0[i] = ya
                y_b0[i] = yb

        n_valid = int((~np.isnan(gap_widths)).sum())
        logger.info(
            f"  GapProfile: ausgewertet in {n_valid}/{self.n_bins} Bins, "
            f"Spaltbreite min={np.nanmin(gap_widths):.2f}, "
            f"max={np.nanmax(gap_widths):.2f} mm"
            if n_valid > 0 else f"  GapProfile: keine validen Bins"
        )

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
            # ── neu: tiefenbezogene Auswertung ────────────────────────
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
            # ── legacy: Extrapolation auf z=0, bis Validierung sauber ─
            "gap_widths": gap_widths if self.emit_legacy_gap_widths else None,
            "y_a_root": y_a0 if self.emit_legacy_gap_widths else None,
            "y_b_root": y_b0 if self.emit_legacy_gap_widths else None,
        }

        artifacts: Dict[str, Any] = {
            "n_bins_total": self.n_bins,
            "n_bins_valid": n_valid,
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

        if n_valid > 0 and self.emit_legacy_gap_widths:
            artifacts.update({
                "gap_min_mm": float(np.nanmin(gap_widths)),
                "gap_max_mm": float(np.nanmax(gap_widths)),
                "gap_mean_mm": float(np.nanmean(gap_widths)),
                "gap_std_mm": float(np.nanstd(gap_widths)),
            })
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

    # ── Helfer ────────────────────────────────────────────────────────

    def _extrapolate_to_z0(self, pts: np.ndarray) -> float:
        """Lineare Anpassung y(z) = m·z + y0, gibt y0 zurück.

        y = pts[:, gap_axis], z = pts[:, vertical_axis].

        Robust gegen Downsampling: der Fit nutzt alle Flankenpunkte des Bins,
        nicht die Randpunkte. Damit fehlt hier der Bias, der
        GapClassifier._compute_gap_width_by_seam() über .max()/.min() trifft
        (dort dokumentiert).

        Genauigkeit dieser Methode selbst: EXAKT. Auf den unregistrierten
        synthetischen T_Y-Scans liefert sie Steigung 1.0000 mit Fehlern
        < 0.001 mm über den gesamten Bereich ty = -1.5 .. +1.5 mm.

        ACHTUNG – die Auswertungshöhe verstärkt Fehler um 2·tan(α):
          Die Extrapolation zielt auf vertical_axis = 0 im Koordinatensystem
          der ÜBERGEBENEN Wolke. Ist die Wolke registriert, ist das die z=0-
          Ebene NACH der Registrierung – ein Registrierungs-Versatz dz
          verschiebt also die Auswertungshöhe.

          Eine V-Naht mit Flankenwinkel α zur Vertikalen öffnet sich mit
          dw/dz = 2·tan(α). Der Verstärkungsfaktor ist damit NAHT-SPEZIFISCH:

              α = 45° (90°-Naht) -> Faktor 2.00      <- aktuelles Bauteil
              α = 30° (60°-Naht) -> Faktor 1.15
              je spitzer die Naht, desto stärker die Verstärkung

          Für das aktuelle Bauteil gilt also:

              Fehler_Spaltbreite = -2 · dz_Registrierung

          Verifiziert über die synthetischen Fälle (Vergleich gegen dieselbe
          Methode auf unregistrierten Rohdaten, wo sie exakt ist):
              translation_y  n=10  Korr 0.99998  Rest-RMS 0.0004 mm
              translation_z  n= 4  Korr 0.99999  Rest-RMS 0.0077 mm
              rotation_x     n=10  Korr 0.99970  Rest-RMS 0.0141 mm
              translation_combo n=5 Korr 0.99978 Rest-RMS 0.0068 mm
          Bei rotation_x erzeugt das Fehler bis 0.94 mm, obwohl der WAHRE
          Spalt sich kaum ändert (1.475..1.528) – die Registrierung allein
          produziert dort die gesamte scheinbare Abweichung.

          rotation_y folgt demselben Mechanismus, aber mit POSITIONSABHÄNGIGEM
          dz: eine Restrotation der Registrierung um die gap_axis kippt die
          Auswertungsebene, der Höhenfehler wächst linear entlang der Naht.
          Mit dz_eff = dz - x̄·sin(ry_reg) statt dz allein:
              ry=0.10°  Vorhersage +0.174  tatsächlich +0.174
              ry=0.25°  Vorhersage +0.435  tatsächlich +0.432
              ry=0.50°  Vorhersage +0.876  tatsächlich +0.849
          Die Deckflächen-Verankerung unten räumt das mit ab: Deckfläche und
          Flanken rotieren gemeinsam, eine daran verankerte Auswertungshöhe
          ist rotationsinvariant.

          AUSNAHME ry >= 1.0°: dort bricht die Registrierung qualitativ ein
          (dz springt von ~0.000 auf -0.244, ICP-Residuum 0.482 – anderes
          lokales Minimum). Modell trifft nicht mehr (+1.425 vs +0.355).
          Dokumentierte Verfahrensgrenze; betrifft auch die Kombis mit
          ry-Anteil (rotation_combo, translation_rotation_combo).

          Ausgeschlossen als Ursache – gemessen, nicht vermutet: weder
          Segmentierung noch Flanken-Paarung. Über ry = 0..1.0° bleiben die
          FlankSegmenter-Kandidatenzahlen stabil (33.545 -> 31.938, -5 %),
          und alle 20 Naht-Bins sehen durchgehend BEIDE Flanken, kein
          einziger einseitiger Slice.

          rotation_z liegt mit max 0.064 mm im Rauschen.

        TODO – Auswertungshöhe an der Geometrie verankern:
          Statt z = 0 die per RANSAC gefittete Deckflächen-Ebene als Bezug
          nehmen (BackgroundRemover legt plane_model und z_center bereits in
          den SegmentationReport). Damit fällt dz vollständig heraus.

          ABER – die Abhängigkeit wird damit verlagert, nicht beseitigt:
          Liegt der Deckflächen-Fit um δ daneben, steht δ an der Stelle von dz
          und geht mit demselben Faktor 2·tan(α) ein. Die Genauigkeit der
          Spaltbreite hängt nach dem Fix also an der QUALITÄT DER DECKFLÄCHEN-
          EBENE – das macht die Robustheit dieses RANSAC-Fits sicherheits-
          kritisch für die Spaltmessung.

          Der Gewinn ist trotzdem real: die Deckfläche ist dicht besetzt und
          gut konditioniert, anders als die rx-Rotation, die ICP schlecht
          auflöst. Bei realen Scans sitzen dort aber Spritzer und Reflexionen –
          genau die Störungen, gegen die ein RANSAC-Ebenenfit abgesichert
          werden muss, bevor man sich auf ihn verlässt.

          Schranken – die zweite ERSETZT die erste, sie ergänzt sie nicht:
            - JETZT (z=0-Methode): für die 0.25-mm-Toleranz aus AP2 muss
              dz < ~0.12 mm bleiben. Synthetisch erfüllt (dz < 0.04 mm),
              bei realen Scans offen.
            - NACH dem Fix: dz fällt heraus, das obige Kriterium wird
              hinfällig. Maßgeblich ist dann der Fit-Fehler der Deckflächen-
              Ebene, mit derselben Schranke δ < ~0.12 mm (bei α=45°;
              bei flacheren Nähten entsprechend lockerer).
          Solange die z=0-Methode noch irgendwo als Fallback existiert, bleibt
          der Faktor winkelabhängig und muss bei anderen Öffnungswinkeln
          (z.B. künftige Heidenbluth-Bauteile) neu bestimmt werden.
        """
        if len(pts) < self.min_points_per_bin:
            return float("nan")
        z = pts[:, self.vertical_axis]
        y = pts[:, self.gap_axis]
        try:
            _, y0 = np.polyfit(z, y, 1)
        except (np.linalg.LinAlgError, ValueError):
            return float("nan")
        return float(y0)