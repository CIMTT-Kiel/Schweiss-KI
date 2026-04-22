"""
Tests für das Segmentation-Modul (AP2.1 Phase 3).

Testet:
- Label-Definitionen (labels.py)
- SegmentationStep-Interface (Timing, Report, Integritätsprüfung, enabled/disabled)
- SegmentationReport (Serialisierung, summary, coverage)
- Einzelne Steps (BackgroundRemover, FlankSegmenter, GapClassifier)
- SegmentationPipeline (End-to-End, fill_unlabeled_with_background)
- from_config (YAML-Loading)

Fixtures basieren auf einer synthetischen V-Naht mit bekannter Geometrie
(60° Öffnungswinkel = 30° pro Flanke zur Vertikalen), damit die
Plausibilitätsmetriken (Flankenwinkel, Ebenenfit) gegen Ground Truth
geprüft werden können. Tests mit echten CMM-Daten gehören in ein Notebook
(Validierung), nicht in die Unit-Tests.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from schweiss_ki.segmentation import (
    LABELS,
    NAME_TO_ID,
    UNLABELED,
    BackgroundRemover,
    FlankSegmenter,
    GapClassifier,
    SegmentationPipeline,
    SegmentationReport,
    SegmentationStep,
    SegmentationStepReport,
)


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

def _make_v_groove(
    half_width: float = 30.0,
    length: float = 40.0,
    gap_half_width: float = 1.25,
    flank_angle_deg: float = 30.0,
    n_top: int = 2000,
    n_flank: int = 800,
    n_sub_gap: int = 50,
    noise: float = 0.02,
    seed: int = 42,
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Synthetische V-Naht mit bekannter Geometrie und Ground-Truth-Labels.

    Geometrie:
      - Oberseite links:   z = 0, x ∈ [-half_width, -gap_half_width]
      - Oberseite rechts:  z = 0, x ∈ [+gap_half_width, +half_width]
      - Flanke A (links):  (-gap_half_width, 0) → (0, -depth)
      - Flanke B (rechts): (+gap_half_width, 0) → (0, -depth)
        mit depth = gap_half_width / tan(flank_angle_deg)
      - Sub-Gap-Artefakte: x ∈ [-gap_half_width, gap_half_width], z < -depth

    Flanken-Normalen nach außen (vom Werkstück weg):
      - Flanke A: ( cos α, 0, sin α )
      - Flanke B: (-cos α, 0, sin α )

    Returns:
        (pcd, ground_truth_labels) — pcd hat Points + Normals gesetzt,
        Labels nach Konvention aus segmentation.labels.
    """
    rng = np.random.default_rng(seed)
    alpha_rad = np.deg2rad(flank_angle_deg)
    depth = gap_half_width / np.tan(alpha_rad)

    # ------------------------------------------------------------------
    # Oberseiten (links + rechts), Normale = (0, 0, 1)
    # ------------------------------------------------------------------
    n_top_half = n_top // 2
    x_l = rng.uniform(-half_width, -gap_half_width, n_top_half)
    x_r = rng.uniform(gap_half_width, half_width, n_top - n_top_half)
    y_t = rng.uniform(0.0, length, n_top)
    z_t = rng.normal(0.0, noise, n_top)
    pts_top = np.column_stack([np.concatenate([x_l, x_r]), y_t, z_t])
    nrm_top = np.tile([0.0, 0.0, 1.0], (n_top, 1))
    lbl_top = np.full(n_top, NAME_TO_ID["background"], dtype=np.int8)

    # ------------------------------------------------------------------
    # Flanke A (linke Fase): Parametrisierung über Tiefen-Parameter t
    # ------------------------------------------------------------------
    def _make_flank(sign: float, n: int, normal: np.ndarray, label: int):
        t = rng.uniform(0.0, 1.0, n)
        x_flank = sign * gap_half_width * (1.0 - t)
        z_flank = -depth * t
        y_flank = rng.uniform(0.0, length, n)
        # Rauschen senkrecht zur Ebene
        noise_vec = rng.normal(0.0, noise, n)
        pts = np.column_stack([x_flank, y_flank, z_flank])
        pts += noise_vec[:, None] * normal[None, :]
        nrm = np.tile(normal, (n, 1))
        lbl = np.full(n, label, dtype=np.int8)
        return pts, nrm, lbl

    normal_a = np.array([np.cos(alpha_rad), 0.0, np.sin(alpha_rad)])
    normal_b = np.array([-np.cos(alpha_rad), 0.0, np.sin(alpha_rad)])
    pts_a, nrm_a, lbl_a = _make_flank(-1.0, n_flank, normal_a, NAME_TO_ID["flank_a"])
    pts_b, nrm_b, lbl_b = _make_flank(+1.0, n_flank, normal_b, NAME_TO_ID["flank_b"])

    # ------------------------------------------------------------------
    # Sub-Gap-Artefakte (unterhalb der V-Naht)
    # ------------------------------------------------------------------
    x_sg = rng.uniform(-gap_half_width * 0.5, gap_half_width * 0.5, n_sub_gap)
    y_sg = rng.uniform(0.0, length, n_sub_gap)
    z_sg = rng.uniform(-depth * 2.0, -depth * 1.1, n_sub_gap)
    pts_sg = np.column_stack([x_sg, y_sg, z_sg])
    # Zufällige Normalen — sollen weder als Background noch Flanke erkannt werden
    nrm_sg = rng.normal(0.0, 1.0, (n_sub_gap, 3))
    nrm_sg /= np.linalg.norm(nrm_sg, axis=1, keepdims=True)
    lbl_sg = np.full(n_sub_gap, NAME_TO_ID["sub_gap_artifacts"], dtype=np.int8)

    # ------------------------------------------------------------------
    # Zusammenbau
    # ------------------------------------------------------------------
    points = np.vstack([pts_top, pts_a, pts_b, pts_sg])
    normals = np.vstack([nrm_top, nrm_a, nrm_b, nrm_sg])
    labels = np.concatenate([lbl_top, lbl_a, lbl_b, lbl_sg])

    # Shuffle – Reihenfolge darf für RANSAC keine Rolle spielen
    perm = rng.permutation(len(points))
    points = points[perm]
    normals = normals[perm]
    labels = labels[perm]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd, labels


@pytest.fixture
def v_groove_pcd() -> o3d.geometry.PointCloud:
    """Synthetische V-Naht ohne Ground-Truth (nur PCD)."""
    pcd, _ = _make_v_groove()
    return pcd


@pytest.fixture
def v_groove_with_gt() -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """Synthetische V-Naht inkl. Ground-Truth-Labels."""
    return _make_v_groove()


@pytest.fixture
def simple_plane_pcd() -> o3d.geometry.PointCloud:
    """Einfache horizontale Ebene mit 500 Punkten, Normalen = (0,0,1)."""
    rng = np.random.default_rng(0)
    n = 500
    points = np.column_stack([
        rng.uniform(-10, 10, n),
        rng.uniform(-10, 10, n),
        rng.normal(0.0, 0.01, n),
    ])
    normals = np.tile([0.0, 0.0, 1.0], (n, 1))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


# ---------------------------------------------------------------------------
# Label-Tests
# ---------------------------------------------------------------------------

class TestLabels:
    def test_all_expected_labels_present(self):
        expected = {"background", "flank_a", "flank_b", "gap_region", "sub_gap_artifacts"}
        assert set(LABELS.values()) == expected

    def test_label_ids_are_zero_based_sequential(self):
        assert sorted(LABELS.keys()) == [0, 1, 2, 3, 4]

    def test_name_to_id_is_inverse(self):
        for label_id, name in LABELS.items():
            assert NAME_TO_ID[name] == label_id

    def test_unlabeled_is_minus_one(self):
        assert UNLABELED == -1

    def test_unlabeled_not_in_labels(self):
        assert UNLABELED not in LABELS


# ---------------------------------------------------------------------------
# Report-Datenklassen
# ---------------------------------------------------------------------------

class TestSegmentationStepReport:
    def test_points_assigned(self):
        r = SegmentationStepReport(
            step_name="test", enabled=True,
            unlabeled_before=100, unlabeled_after=30,
            assigned={0: 50, 1: 20},
            duration_ms=1.0,
        )
        assert r.points_assigned == 70

    def test_to_dict_json_serializable(self):
        r = SegmentationStepReport(
            step_name="test", enabled=True,
            unlabeled_before=100, unlabeled_after=50,
            assigned={0: 30, 1: 20},
            duration_ms=1.5,
            params={"threshold": 0.25},
            artifacts={"plane_normal": np.array([0.0, 0.0, 1.0])},
        )
        d = r.to_dict()
        # Muss json-serialisierbar sein
        json.dumps(d)
        assert d["step_name"] == "test"
        assert d["assigned"] == {"0": 30, "1": 20}
        assert d["artifacts"]["plane_normal"] == [0.0, 0.0, 1.0]

    def test_repr_contains_step_name(self):
        r = SegmentationStepReport(
            step_name="my_step", enabled=True,
            unlabeled_before=100, unlabeled_after=50,
            assigned={0: 50},
            duration_ms=1.0,
        )
        assert "my_step" in repr(r)


class TestSegmentationReport:
    def test_coverage_pct_all_labeled(self):
        r = SegmentationReport(
            n_points=1000,
            points_per_label={0: 500, 1: 300, 3: 200},
            total_duration_ms=10.0,
        )
        assert r.coverage_pct == pytest.approx(100.0)

    def test_coverage_pct_partial(self):
        r = SegmentationReport(
            n_points=1000,
            points_per_label={UNLABELED: 200, 0: 800},
            total_duration_ms=10.0,
        )
        assert r.coverage_pct == pytest.approx(80.0)

    def test_coverage_pct_empty(self):
        r = SegmentationReport(n_points=0, points_per_label={}, total_duration_ms=0.0)
        assert r.coverage_pct == 100.0

    def test_artifacts_of_missing_step(self):
        r = SegmentationReport(n_points=100, points_per_label={0: 100}, total_duration_ms=1.0)
        assert r.artifacts_of("nonexistent") == {}

    def test_summary_contains_all_label_names(self):
        r = SegmentationReport(
            n_points=100,
            points_per_label={0: 50, 1: 25, 2: 25},
            total_duration_ms=1.0,
        )
        summary = r.summary()
        assert "background" in summary
        assert "flank_a" in summary
        assert "flank_b" in summary

    def test_to_dict_json_serializable(self):
        r = SegmentationReport(
            n_points=100,
            points_per_label={0: 50, 1: 50},
            total_duration_ms=1.0,
        )
        json.dumps(r.to_dict())


# ---------------------------------------------------------------------------
# SegmentationStep – Basis-Invarianten
# ---------------------------------------------------------------------------

class _TestStep(SegmentationStep):
    """Minimaler Test-Step, klassifiziert alle UNLABELED als 'target_label'."""

    def __init__(self, target_label: int = 0, enabled: bool = True):
        self._target_label = target_label
        self._enabled = enabled

    @property
    def name(self) -> str:
        return "test_step"

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _apply(self, pcd, labels):
        labels_out = labels.copy()
        labels_out[labels == UNLABELED] = self._target_label
        return labels_out

    def get_params(self) -> dict:
        return {"target_label": self._target_label}


class _OverwritingStep(SegmentationStep):
    """Böser Step – überschreibt bestehende Labels. Soll RuntimeError triggern."""

    @property
    def name(self) -> str:
        return "overwriting_step"

    @property
    def enabled(self) -> bool:
        return True

    def _apply(self, pcd, labels):
        labels_out = labels.copy()
        labels_out[:] = 99  # überschreibt ALLES
        return labels_out

    def get_params(self) -> dict:
        return {}


class TestSegmentationStepBase:
    def test_disabled_passes_labels_through(self, simple_plane_pcd):
        labels = np.full(len(simple_plane_pcd.points), UNLABELED, dtype=np.int8)
        step = _TestStep(enabled=False)
        labels_out, report = step.apply(simple_plane_pcd, labels)
        assert np.array_equal(labels_out, labels)
        assert report.enabled is False
        assert report.points_assigned == 0

    def test_enabled_step_assigns_all_unlabeled(self, simple_plane_pcd):
        labels = np.full(len(simple_plane_pcd.points), UNLABELED, dtype=np.int8)
        step = _TestStep(target_label=0)
        labels_out, report = step.apply(simple_plane_pcd, labels)
        assert (labels_out == 0).all()
        assert report.points_assigned == len(simple_plane_pcd.points)

    def test_overwriting_step_raises(self, simple_plane_pcd):
        labels = np.full(len(simple_plane_pcd.points), 0, dtype=np.int8)
        step = _OverwritingStep()
        with pytest.raises(RuntimeError, match="überschrieben"):
            step.apply(simple_plane_pcd, labels)

    def test_timing_recorded(self, simple_plane_pcd):
        labels = np.full(len(simple_plane_pcd.points), UNLABELED, dtype=np.int8)
        _, report = _TestStep().apply(simple_plane_pcd, labels)
        assert report.duration_ms >= 0.0

    def test_report_assigned_tracks_new_labels(self, simple_plane_pcd):
        n = len(simple_plane_pcd.points)
        labels = np.full(n, UNLABELED, dtype=np.int8)
        labels[:100] = 2  # 100 Punkte schon Flank B – dürfen nicht gezählt werden
        step = _TestStep(target_label=0)
        _, report = step.apply(simple_plane_pcd, labels)
        # Nur die restlichen n-100 Punkte wurden in diesem Step zugewiesen
        assert report.assigned == {0: n - 100}


# ---------------------------------------------------------------------------
# BackgroundRemover
# ---------------------------------------------------------------------------

class TestBackgroundRemover:
    def test_requires_normals(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))
        labels = np.full(100, UNLABELED, dtype=np.int8)
        with pytest.raises(ValueError, match="Normalen"):
            BackgroundRemover()._apply(pcd, labels)

    def test_fits_horizontal_top(self, v_groove_with_gt):
        pcd, gt = v_groove_with_gt
        labels = np.full(len(pcd.points), UNLABELED, dtype=np.int8)
        step = BackgroundRemover()
        labels_out, report = step.apply(pcd, labels)

        # Tilt nahe 0 für horizontale Ebene
        assert report.artifacts["tilt_angle_deg"] < 5.0
        # Plane-Normal ≈ [0, 0, 1]
        n_plane = np.array(report.artifacts["plane_normal"])
        assert abs(n_plane[2]) > 0.99

    def test_classifies_top_surface_as_background(self, v_groove_with_gt):
        pcd, gt = v_groove_with_gt
        labels = np.full(len(pcd.points), UNLABELED, dtype=np.int8)
        step = BackgroundRemover()
        labels_out, _ = step.apply(pcd, labels)

        # Mindestens 80% der echten Background-Punkte sollten erkannt werden
        gt_bg = gt == NAME_TO_ID["background"]
        recall = (labels_out[gt_bg] == NAME_TO_ID["background"]).mean()
        assert recall > 0.8

        # Flanken-Punkte am Übergang zur Oberseite (obere ~5% der Flanke)
        # liegen geometrisch innerhalb der ransac_threshold-Toleranz und
        # können korrekterweise als Background klassifiziert werden.
        gt_flank = (gt == NAME_TO_ID["flank_a"]) | (gt == NAME_TO_ID["flank_b"])
        false_pos_rate = (labels_out[gt_flank] == NAME_TO_ID["background"]).mean()
        assert false_pos_rate < 0.15

    def test_disabled_no_change(self, v_groove_pcd):
        labels = np.full(len(v_groove_pcd.points), UNLABELED, dtype=np.int8)
        step = BackgroundRemover(enabled=False)
        labels_out, report = step.apply(v_groove_pcd, labels)
        assert (labels_out == UNLABELED).all()
        assert report.enabled is False

    def test_no_unlabeled_points_skips_gracefully(self, simple_plane_pcd):
        labels = np.full(len(simple_plane_pcd.points), 0, dtype=np.int8)
        step = BackgroundRemover()
        labels_out, report = step.apply(simple_plane_pcd, labels)
        assert np.array_equal(labels_out, labels)
        assert report.artifacts == {}

    def test_params_in_report(self, simple_plane_pcd):
        labels = np.full(len(simple_plane_pcd.points), UNLABELED, dtype=np.int8)
        step = BackgroundRemover(ransac_threshold=0.1, normal_cos_threshold=0.9)
        _, report = step.apply(simple_plane_pcd, labels)
        assert report.params["ransac_threshold"] == pytest.approx(0.1)
        assert report.params["normal_cos_threshold"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# FlankSegmenter
# ---------------------------------------------------------------------------

class TestFlankSegmenter:
    def test_requires_normals(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))
        labels = np.full(100, UNLABELED, dtype=np.int8)
        with pytest.raises(ValueError, match="Normalen"):
            FlankSegmenter()._apply(pcd, labels)

    def test_fits_both_flanks_at_expected_angle(self, v_groove_with_gt):
        pcd, _ = v_groove_with_gt
        labels = np.full(len(pcd.points), UNLABELED, dtype=np.int8)
        step = FlankSegmenter(expected_flank_angle_deg=30.0)
        _, report = step.apply(pcd, labels)

        a = report.artifacts["flank_a"]
        b = report.artifacts["flank_b"]
        assert a["status"] == "ok"
        assert b["status"] == "ok"
        assert abs(a["angle_from_vertical_deg"] - 30.0) < 3.0
        assert abs(b["angle_from_vertical_deg"] - 30.0) < 3.0

    def test_flank_a_left_flank_b_right(self, v_groove_with_gt):
        """Flank A = linke Fase (n_x > 0), Flank B = rechte Fase (n_x < 0)."""
        pcd, _ = v_groove_with_gt
        labels = np.full(len(pcd.points), UNLABELED, dtype=np.int8)
        step = FlankSegmenter()
        _, report = step.apply(pcd, labels)
        n_a = np.array(report.artifacts["flank_a"]["plane_normal"])
        n_b = np.array(report.artifacts["flank_b"]["plane_normal"])
        assert n_a[0] > 0, f"Flank A Normale sollte n_x > 0 haben, ist {n_a}"
        assert n_b[0] < 0, f"Flank B Normale sollte n_x < 0 haben, ist {n_b}"

    def test_classifies_flank_points_correctly(self, v_groove_with_gt):
        pcd, gt = v_groove_with_gt
        labels = np.full(len(pcd.points), UNLABELED, dtype=np.int8)
        step = FlankSegmenter()
        labels_out, _ = step.apply(pcd, labels)

        for side in ("flank_a", "flank_b"):
            gt_mask = gt == NAME_TO_ID[side]
            recall = (labels_out[gt_mask] == NAME_TO_ID[side]).mean()
            assert recall > 0.7, f"{side} recall zu niedrig: {recall:.2f}"

    def test_insufficient_candidates_reports_status(self, simple_plane_pcd):
        # Horizontale Ebene hat keine Kandidaten für 30°-Flanken
        labels = np.full(len(simple_plane_pcd.points), UNLABELED, dtype=np.int8)
        step = FlankSegmenter(normal_cos_threshold=0.95)
        _, report = step.apply(simple_plane_pcd, labels)
        assert report.artifacts["flank_a"]["status"] == "insufficient_candidates"
        assert report.artifacts["flank_b"]["status"] == "insufficient_candidates"

    def test_disabled_no_change(self, v_groove_pcd):
        labels = np.full(len(v_groove_pcd.points), UNLABELED, dtype=np.int8)
        step = FlankSegmenter(enabled=False)
        labels_out, _ = step.apply(v_groove_pcd, labels)
        assert (labels_out == UNLABELED).all()


# ---------------------------------------------------------------------------
# GapClassifier
# ---------------------------------------------------------------------------

class TestGapClassifier:
    def test_skips_without_flanks(self, v_groove_pcd):
        labels = np.full(len(v_groove_pcd.points), UNLABELED, dtype=np.int8)
        step = GapClassifier()
        labels_out, report = step.apply(v_groove_pcd, labels)
        assert (labels_out == UNLABELED).all()
        assert report.artifacts == {}

    def test_separates_sub_gap_artifacts(self, v_groove_with_gt):
        """Artefakte unter der V-Naht → Label 4 (wenn separate=True)."""
        pcd, gt = v_groove_with_gt
        # Ground-Truth-Flanken + UNLABELED für Rest (inkl. Sub-Gap)
        labels = np.where(
            (gt == NAME_TO_ID["flank_a"]) | (gt == NAME_TO_ID["flank_b"]),
            gt,
            UNLABELED,
        ).astype(np.int8)

        step = GapClassifier(separate_sub_gap_artifacts=True)
        labels_out, report = step.apply(pcd, labels)

        # Mindestens ein Teil der echten Sub-Gap-Punkte muss als Label 4 klassifiziert sein
        gt_sub_gap = gt == NAME_TO_ID["sub_gap_artifacts"]
        recall = (labels_out[gt_sub_gap] == NAME_TO_ID["sub_gap_artifacts"]).mean()
        assert recall > 0.5
        assert report.artifacts["n_sub_gap"] > 0

    def test_sub_gap_merged_into_background_when_disabled(self, v_groove_with_gt):
        pcd, gt = v_groove_with_gt
        labels = np.where(
            (gt == NAME_TO_ID["flank_a"]) | (gt == NAME_TO_ID["flank_b"]),
            gt,
            UNLABELED,
        ).astype(np.int8)

        step = GapClassifier(separate_sub_gap_artifacts=False)
        labels_out, _ = step.apply(pcd, labels)

        # Keine Punkte mit Label sub_gap_artifacts (4)
        assert (labels_out == NAME_TO_ID["sub_gap_artifacts"]).sum() == 0
        # Die Sub-Gap-Punkte müssen Background sein
        gt_sub_gap = gt == NAME_TO_ID["sub_gap_artifacts"]
        assert (labels_out[gt_sub_gap] == NAME_TO_ID["background"]).any()

    def test_z_bounds_in_artifacts(self, v_groove_with_gt):
        pcd, gt = v_groove_with_gt
        labels = np.where(
            (gt == NAME_TO_ID["flank_a"]) | (gt == NAME_TO_ID["flank_b"]),
            gt,
            UNLABELED,
        ).astype(np.int8)

        step = GapClassifier()
        _, report = step.apply(pcd, labels)
        assert report.artifacts["z_upper"] > report.artifacts["z_lower"]
        assert report.artifacts["x_min"] < report.artifacts["x_max"]

    def test_gap_width_by_y_structure(self, v_groove_with_gt):
        pcd, gt = v_groove_with_gt
        labels = np.where(
            (gt == NAME_TO_ID["flank_a"]) | (gt == NAME_TO_ID["flank_b"]),
            gt,
            UNLABELED,
        ).astype(np.int8)

        step = GapClassifier(gap_width_y_bins=10)
        _, report = step.apply(pcd, labels)
        widths = report.artifacts["gap_width_by_y"]
        assert isinstance(widths, list)
        if widths:
            assert all(len(row) == 2 for row in widths)

    def test_disabled_no_change(self, v_groove_with_gt):
        pcd, gt = v_groove_with_gt
        labels = gt.copy().astype(np.int8)
        step = GapClassifier(enabled=False)
        labels_out, _ = step.apply(pcd, labels)
        assert np.array_equal(labels_out, labels)


# ---------------------------------------------------------------------------
# SegmentationPipeline
# ---------------------------------------------------------------------------

class TestSegmentationPipeline:
    def test_empty_pipeline(self, v_groove_pcd):
        pipeline = SegmentationPipeline()
        labels, report = pipeline.process(v_groove_pcd)
        # Alle UNLABELED → Background (fill_unlabeled_with_background=True default)
        assert (labels == 0).all()
        assert report.points_per_label.get(0) == len(v_groove_pcd.points)

    def test_fill_unlabeled_disabled_keeps_minus_one(self, v_groove_pcd):
        pipeline = SegmentationPipeline(fill_unlabeled_with_background=False)
        labels, _ = pipeline.process(v_groove_pcd)
        assert (labels == UNLABELED).all()

    def test_end_to_end_v_groove(self, v_groove_with_gt):
        pcd, gt = v_groove_with_gt
        pipeline = SegmentationPipeline([
            BackgroundRemover(),
            FlankSegmenter(),
            GapClassifier(),
        ])
        labels, report = pipeline.process(pcd)

        # Kein UNLABELED mehr nach fill
        assert (labels == UNLABELED).sum() == 0
        # Alle 3 Kern-Klassen vorhanden
        assert report.points_per_label.get(NAME_TO_ID["background"], 0) > 0
        assert report.points_per_label.get(NAME_TO_ID["flank_a"], 0) > 0
        assert report.points_per_label.get(NAME_TO_ID["flank_b"], 0) > 0
        # Summe stimmt
        assert sum(report.points_per_label.values()) == len(pcd.points)
        # Coverage 100% durch fill
        assert report.coverage_pct == pytest.approx(100.0)

    def test_end_to_end_plausibility_vs_gt(self, v_groove_with_gt):
        """Overall Accuracy der Pipeline gegen Ground-Truth."""
        pcd, gt = v_groove_with_gt
        pipeline = SegmentationPipeline([
            BackgroundRemover(),
            FlankSegmenter(),
            GapClassifier(separate_sub_gap_artifacts=True),
        ])
        labels, _ = pipeline.process(pcd)
        accuracy = (labels == gt).mean()
        # Auf synthetischen Daten sollte Accuracy hoch sein
        assert accuracy > 0.85, f"Accuracy zu niedrig: {accuracy:.3f}"

    def test_add_step_chainable(self):
        pipeline = SegmentationPipeline()
        result = pipeline.add_step(BackgroundRemover())
        assert result is pipeline
        assert len(pipeline.steps) == 1

    def test_remove_step(self):
        pipeline = SegmentationPipeline([BackgroundRemover(), FlankSegmenter()])
        removed = pipeline.remove_step("background_remover")
        assert removed is True
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].name == "flank_segmenter"

    def test_remove_nonexistent_step(self):
        pipeline = SegmentationPipeline([BackgroundRemover()])
        assert pipeline.remove_step("nonexistent") is False
        assert len(pipeline.steps) == 1

    def test_repr_contains_step_names(self):
        pipeline = SegmentationPipeline([BackgroundRemover(), FlankSegmenter()])
        r = repr(pipeline)
        assert "background_remover" in r
        assert "flank_segmenter" in r


# ---------------------------------------------------------------------------
# from_config (YAML-Loading)
# ---------------------------------------------------------------------------

class TestFromConfig:
    def test_loads_full_pipeline(self, tmp_path: Path):
        config = """
segmentation:
  enabled: true
  steps:
    background_remover:
      enabled: true
      ransac_threshold: 0.3
    flank_segmenter:
      enabled: true
      expected_flank_angle_deg: 30.0
    gap_classifier:
      enabled: true
      x_margin: 0.75
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config)

        pipeline = SegmentationPipeline.from_config(config_path)
        step_names = [s.name for s in pipeline.steps]
        assert step_names == ["background_remover", "flank_segmenter", "gap_classifier"]

    def test_disabled_step_is_skipped(self, tmp_path: Path):
        config = """
segmentation:
  steps:
    background_remover:
      enabled: true
    flank_segmenter:
      enabled: false
    gap_classifier:
      enabled: true
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config)

        pipeline = SegmentationPipeline.from_config(config_path)
        step_names = [s.name for s in pipeline.steps]
        assert "flank_segmenter" not in step_names
        assert len(pipeline.steps) == 2

    def test_params_forwarded(self, tmp_path: Path):
        config = """
segmentation:
  steps:
    background_remover:
      enabled: true
      ransac_threshold: 0.42
      normal_cos_threshold: 0.88
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config)

        pipeline = SegmentationPipeline.from_config(config_path)
        step = pipeline.steps[0]
        params = step.get_params()
        assert params["ransac_threshold"] == pytest.approx(0.42)
        assert params["normal_cos_threshold"] == pytest.approx(0.88)

    def test_unknown_step_raises(self, tmp_path: Path):
        config = """
segmentation:
  steps:
    bogus_step:
      enabled: true
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config)

        with pytest.raises(ValueError, match="Unbekannter Segmentation-Step"):
            SegmentationPipeline.from_config(config_path)

    def test_empty_config_gives_empty_pipeline(self, tmp_path: Path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("segmentation:\n  steps: {}\n")
        pipeline = SegmentationPipeline.from_config(config_path)
        assert len(pipeline.steps) == 0