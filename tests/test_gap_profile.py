"""
Tests für GapProfile (AP2.2) – tiefenbezogene Spaltmessung.

Schwerpunkt sind die beiden Regressionstests, die den Verstärkungs-
mechanismus festnageln:

1. dz-Invarianz (TestDepthAnchoring)
   Die Auswertungshöhe ist an der Deckflächen-Ebene des Referenz-Werkstücks
   verankert, nicht an vertical_axis = 0. Ein Registrierungs-Höhenversatz dz
   darf die gemessene Spaltbreite deshalb NICHT verändern.

   Vor dem Fix ging dz mit Faktor dw/dz = 2·tan(α) in die Spaltbreite ein –
   bei der 90°-Naht also verdoppelt. Auf den 61 synthetischen Fällen war das
   die alleinige Ursache der verbliebenen Untererfassung (Korrelation 0.99998
   zwischen Fehler und -2·dz). Dieser Test hält den Fix ohne den Datensatz
   fest und wuerde einen Rueckfall sofort zeigen.

2. Robustheit der Deckflächen-Ebene (TestTopPlaneRobustness)
   Der Faktor 2·tan(α) gilt auch für Fehler der Referenzebene SELBST. Bei
   realen Scans sitzen Spritzer und Reflexionen genau dort. Getestet wird,
   dass der RANSAC-Fit sie verträgt, dass inlier_ratio die Störung anzeigt
   und dass das Gate bei zu starker Kontamination greift.

Die Fixture hat – anders als die V-Naht in test_segmentation.py – eine
definierte Wurzelöffnung, damit der gemessene Spalt gegen einen bekannten
Wert prüfbar ist.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from schweiss_ki.subtraction.deviation.gap_profile import GapProfile
from schweiss_ki.subtraction.reports import DeviationData

BACKGROUND, FLANK_A, FLANK_B = 0, 1, 2


def make_v_seam(
    root_gap: float = 2.0,
    thickness: float = 5.0,
    flank_angle_deg: float = 45.0,
    seam_length: float = 60.0,
    half_width: float = 25.0,
    n_top: int = 12000,
    n_flank: int = 6000,
    noise: float = 0.005,
    seed: int = 7,
):
    """V-Naht mit definierter Wurzelöffnung.

    Koordinaten: seam = X, gap = Y, vertical = Z.
    Deckfläche bei z = 0, Wurzel bei z = -thickness.

    Flanke A (negative gap-Seite), als Funktion der Tiefe d unter der
    Deckfläche:
        y_A(d) = -root_gap/2 - (thickness - d)·tan(α)
    Flanke B spiegelbildlich. Daraus:
        gap(d) = root_gap + 2·(thickness - d)·tan(α)
    An der Wurzel (d = thickness) also exakt root_gap.

    Returns:
        (pcd, labels) – pcd mit Punkten, labels nach AP2.1-Konvention.
    """
    rng = np.random.default_rng(seed)
    tan_a = np.tan(np.deg2rad(flank_angle_deg))
    y_top_edge = root_gap / 2 + thickness * tan_a  # Fasenkante an der Oberseite

    # ── Deckflächen beidseits ────────────────────────────────────────
    n_half = n_top // 2
    y_pos = rng.uniform(y_top_edge, half_width, n_half)
    y_neg = rng.uniform(-half_width, -y_top_edge, n_top - n_half)
    x_top = rng.uniform(0.0, seam_length, n_top)
    z_top = rng.normal(0.0, noise, n_top)
    pts_top = np.column_stack([x_top, np.concatenate([y_pos, y_neg]), z_top])
    lbl_top = np.full(n_top, BACKGROUND, dtype=np.int8)

    # ── Flanken ──────────────────────────────────────────────────────
    def flank(sign: float, label: int):
        d = rng.uniform(0.0, thickness, n_flank)          # Tiefe unter Deckfläche
        y = sign * (root_gap / 2 + (thickness - d) * tan_a)
        x = rng.uniform(0.0, seam_length, n_flank)
        pts = np.column_stack([x, y, -d])
        pts[:, 1] += rng.normal(0.0, noise, n_flank)
        return pts, np.full(n_flank, label, dtype=np.int8)

    pts_a, lbl_a = flank(-1.0, FLANK_A)
    pts_b, lbl_b = flank(+1.0, FLANK_B)

    points = np.vstack([pts_top, pts_a, pts_b])
    labels = np.concatenate([lbl_top, lbl_a, lbl_b])
    perm = rng.permutation(len(points))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[perm])
    return pcd, labels[perm]


@pytest.fixture
def v_seam():
    return make_v_seam()


@pytest.fixture
def step():
    return GapProfile(seam_axis=0, gap_axis=1, vertical_axis=2)


def measure(step: GapProfile, pcd, labels) -> dict:
    """GapProfile ausführen, Artefakte + DeviationData zurückgeben."""
    data = DeviationData()
    art = step._apply(pcd, pcd, data, labels, None)
    return {"artifacts": art, "profile": data.gap_profile}


# ---------------------------------------------------------------------------
# 1. dz-Invarianz – der zentrale Regressionstest
# ---------------------------------------------------------------------------

class TestDepthAnchoring:
    def test_measures_root_gap(self, step, v_seam):
        """Grundfall: der gemessene Wurzelspalt trifft die Fixture-Geometrie."""
        pcd, lbl = v_seam
        r = measure(step, pcd, lbl)
        a = r["artifacts"]
        assert a["anchored"] is True
        assert a["n_bins_valid_depth"] > 0

        # gap(d) = root_gap + 2·(thickness - d)·tan(45°); ausgewertet bei d_root
        d_root = float(np.nanmedian(r["profile"]["d_root"]))
        erwartet = 2.0 + 2.0 * (5.0 - d_root)
        assert a["gap_root_mean_mm"] == pytest.approx(erwartet, abs=0.05)

    def test_flank_angle_is_measured_not_assumed(self, step):
        """Der Flankenwinkel wird gemessen – auch wenn er vom Soll abweicht."""
        for soll in (30.0, 45.0):
            pcd, lbl = make_v_seam(flank_angle_deg=soll)
            a = measure(step, pcd, lbl)["artifacts"]
            assert a["flank_a_angle_deg"] == pytest.approx(soll, abs=0.5)
            assert a["flank_b_angle_deg"] == pytest.approx(soll, abs=0.5)
            assert a["flank_angle_asymmetry_deg"] < 0.5

    @pytest.mark.parametrize("dz", [-2.5, -1.0, -0.2, 0.2, 1.0, 2.5])
    def test_gap_is_invariant_under_vertical_shift(self, step, v_seam, dz):
        """Kern des Fixes: ein Höhenversatz darf die Spaltbreite nicht ändern.

        Vor der Verankerung ging dz mit -2·dz in die Spaltbreite ein. Schlägt
        dieser Test fehl, ist der Faktor-2-Fehler zurück.
        """
        pcd, lbl = v_seam
        ref = measure(step, pcd, lbl)["artifacts"]["gap_root_mean_mm"]

        shifted = o3d.geometry.PointCloud(pcd)
        shifted.translate((0.0, 0.0, dz))
        got = measure(step, shifted, lbl)["artifacts"]["gap_root_mean_mm"]

        assert got == pytest.approx(ref, abs=1e-6), (
            f"Spaltbreite haengt von dz ab: {ref:.6f} -> {got:.6f} bei dz={dz}. "
            f"Erwartet waere -2*dz = {-2*dz:+.3f} mm, wenn die Verankerung fehlt."
        )

    def test_legacy_method_does_depend_on_dz(self, step, v_seam):
        """Gegenprobe: die alte z=0-Methode zeigt den -2·dz-Effekt.

        Stellt sicher, dass der Invarianztest oben ueberhaupt etwas pruefen
        kann und nicht auf einer Fixture laeuft, bei der dz folgenlos waere.
        """
        pcd, lbl = v_seam
        base = measure(step, pcd, lbl)["artifacts"]["gap_mean_mm"]
        dz = 0.3
        shifted = o3d.geometry.PointCloud(pcd)
        shifted.translate((0.0, 0.0, dz))
        got = measure(step, shifted, lbl)["artifacts"]["gap_mean_mm"]
        assert got == pytest.approx(base - 2.0 * dz, abs=0.02)

    def test_invariant_under_rigid_tilt(self, step, v_seam):
        """Auch gegen eine Starrkörper-Verkippung invariant.

        Deckfläche und Flanken drehen gemeinsam; eine daran verankerte
        Auswertungshöhe ist rotationsinvariant. Das ist der Grund, warum der
        Fix auch die rotation_x/rotation_y-Faelle abraeumt.
        """
        pcd, lbl = v_seam
        ref = measure(step, pcd, lbl)["artifacts"]["gap_root_mean_mm"]
        tilted = o3d.geometry.PointCloud(pcd)
        R = tilted.get_rotation_matrix_from_xyz((np.deg2rad(1.0), 0.0, 0.0))
        tilted.rotate(R, center=(0.0, 0.0, 0.0))
        got = measure(step, tilted, lbl)["artifacts"]["gap_root_mean_mm"]
        assert got == pytest.approx(ref, abs=0.02)


# ---------------------------------------------------------------------------
# 2. Robustheit der Deckflächen-Ebene
# ---------------------------------------------------------------------------

def add_spatter(pcd, labels, fraction: float, height: float = 1.5, seed: int = 3):
    """Setzt Ausreißer auf die REFERENZ-Deckfläche (positive gap-Seite).

    Simuliert Spritzer/Reflexionen: Punkte mit Background-Label, die deutlich
    ueber der echten Deckflaeche liegen.
    """
    rng = np.random.default_rng(seed)
    pts = np.asarray(pcd.points).copy()
    ref_mask = (labels == BACKGROUND) & (pts[:, 1] >= 0)
    idx = np.where(ref_mask)[0]
    n = int(len(idx) * fraction)
    if n:
        pick = rng.choice(idx, n, replace=False)
        pts[pick, 2] += rng.uniform(0.5 * height, height, n)
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts)
    return out, labels


class TestTopPlaneRobustness:
    def test_clean_surface_has_full_inlier_ratio(self, step, v_seam):
        pcd, lbl = v_seam
        a = measure(step, pcd, lbl)["artifacts"]
        assert a["reference_plane_inlier_ratio"] > 0.98
        assert a["reference_plane_rms_mm"] < 0.05

    @pytest.mark.parametrize("fraction", [0.05, 0.15, 0.30])
    def test_ransac_tolerates_spatter(self, step, v_seam, fraction):
        """RANSAC haelt den Fit, solange die Stoerung Minderheit bleibt.

        Ein Least-Squares-Fit wuerde hier mitwandern; der Fehler ginge mit
        Faktor 2 in die Spaltbreite ein.
        """
        pcd, lbl = v_seam
        ref = measure(step, pcd, lbl)["artifacts"]["gap_root_mean_mm"]
        dirty_pcd, dirty_lbl = add_spatter(pcd, lbl, fraction)
        a = measure(step, dirty_pcd, dirty_lbl)["artifacts"]

        assert a["anchored"] is True
        # inlier_ratio zeigt die Stoerung an
        assert a["reference_plane_inlier_ratio"] < 1.0 - fraction / 2
        # Spaltbreite bleibt innerhalb der AP2-Toleranz
        assert a["gap_root_mean_mm"] == pytest.approx(ref, abs=0.25), (
            f"Spritzer-Anteil {fraction:.0%} verschiebt die Spaltbreite um "
            f"{a['gap_root_mean_mm'] - ref:+.3f} mm"
        )

    def test_inlier_ratio_tracks_contamination(self, step, v_seam):
        """inlier_ratio muss monoton mit dem Stoeranteil fallen – sonst taugt
        es nicht als Guetemass fuer reale Scans."""
        pcd, lbl = v_seam
        ratios = []
        for f in (0.0, 0.10, 0.25, 0.40):
            d_pcd, d_lbl = add_spatter(pcd, lbl, f)
            a = measure(step, d_pcd, d_lbl)["artifacts"]
            ratios.append(a["reference_plane_inlier_ratio"])
        assert all(b <= a + 1e-9 for a, b in zip(ratios, ratios[1:])), ratios
        assert ratios[0] - ratios[-1] > 0.2

    def test_gate_blocks_when_surface_too_dirty(self, v_seam):
        """Bei zu starker Kontamination wird die Verankerung verweigert,
        statt einen falschen Bezug zu liefern."""
        pcd, lbl = v_seam
        strict = GapProfile(
            seam_axis=0, gap_axis=1, vertical_axis=2,
            top_plane_min_inlier_ratio=0.95,
        )
        dirty_pcd, dirty_lbl = add_spatter(pcd, lbl, 0.30)
        a = measure(strict, dirty_pcd, dirty_lbl)["artifacts"]
        assert a["anchored"] is False
        assert "gap_root_mean_mm" not in a

    def test_relative_pose_detects_kantenversatz(self, step, v_seam):
        """Höhenversatz der Gegenseite wird als eigenes Merkmal ausgegeben,
        nicht in die Referenzhöhe eingemittelt."""
        pcd, lbl = v_seam
        pts = np.asarray(pcd.points).copy()
        offset = 0.4
        pts[(lbl == BACKGROUND) & (pts[:, 1] < 0), 2] += offset
        moved = o3d.geometry.PointCloud()
        moved.points = o3d.utility.Vector3dVector(pts)

        r = measure(step, moved, lbl)
        a, rel = r["artifacts"], r["profile"]["opposite_vs_reference"]
        assert rel is not None
        assert a["kantenversatz_mm"] == pytest.approx(offset, abs=0.05)
        # Referenzebene bleibt unbeeinflusst -> Spaltbreite unveraendert
        clean = measure(step, pcd, lbl)["artifacts"]["gap_root_mean_mm"]
        assert a["gap_root_mean_mm"] == pytest.approx(clean, abs=0.02)
