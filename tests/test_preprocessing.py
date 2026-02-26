"""
Tests für das Preprocessing-Modul (AP2.1 Phase 2).

Testet:
- PreprocessingStep-Interface (Timing, Report-Erzeugung, enabled/disabled)
- PreprocessingReport (Serialisierung, summary)
- PreprocessingPipeline (Ausführung, Verkettung)
- Einzelne Steps (StatisticalOutlierFilter, VoxelGridDownsampler, NormalEstimator, ...)
- Integration mit WeldVolumeModel
"""
import numpy as np
import open3d as o3d
import pytest

from schweiss_ki.preprocessing import (
    NormalEstimator,
    PreprocessingPipeline,
    PreprocessingReport,
    RadiusOutlierFilter,
    RandomDownsampler,
    StatisticalOutlierFilter,
    VoxelGridDownsampler,
)
from schweiss_ki.preprocessing.base import StepReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_sphere_pcd() -> o3d.geometry.PointCloud:
    """Saubere Kugel-Punktwolke ohne Rauschen (1000 Punkte, Radius 50mm)."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=50.0)
    pcd = mesh.sample_points_uniformly(number_of_points=1000)
    return pcd


@pytest.fixture
def noisy_sphere_pcd(clean_sphere_pcd) -> o3d.geometry.PointCloud:
    """Kugel-Punktwolke mit Gaussschem Rauschen und 50 Ausreißern."""
    pcd = clean_sphere_pcd
    points = np.asarray(pcd.points)

    # Gauss-Rauschen
    noise = np.random.default_rng(42).normal(0, 0.5, points.shape)
    points += noise

    # Ausreißer weit außerhalb der Kugel
    outlier_count = 50
    outliers = np.random.default_rng(0).uniform(-200, 200, (outlier_count, 3))
    all_points = np.vstack([points, outliers])

    noisy_pcd = o3d.geometry.PointCloud()
    noisy_pcd.points = o3d.utility.Vector3dVector(all_points)
    return noisy_pcd


@pytest.fixture
def large_pcd() -> o3d.geometry.PointCloud:
    """Große Punktwolke für Downsampling-Tests (10.000 Punkte)."""
    mesh = o3d.geometry.TriangleMesh.create_box(width=100, height=50, depth=20)
    pcd = mesh.sample_points_uniformly(number_of_points=10_000)
    return pcd


# ---------------------------------------------------------------------------
# StepReport Tests
# ---------------------------------------------------------------------------

class TestStepReport:
    def test_points_removed(self):
        r = StepReport("test", True, 1000, 800, 10.0)
        assert r.points_removed == 200

    def test_retention_pct(self):
        r = StepReport("test", True, 1000, 500, 10.0)
        assert r.retention_pct == pytest.approx(50.0)

    def test_retention_pct_zero_input(self):
        r = StepReport("test", True, 0, 0, 0.0)
        assert r.retention_pct == 100.0


# ---------------------------------------------------------------------------
# PreprocessingReport Tests
# ---------------------------------------------------------------------------

class TestPreprocessingReport:
    def test_points_removed(self):
        r = PreprocessingReport(input_points=1000, output_points=700, total_duration_ms=50.0)
        assert r.points_removed == 300

    def test_overall_retention_pct(self):
        r = PreprocessingReport(input_points=1000, output_points=800, total_duration_ms=50.0)
        assert r.overall_retention_pct == pytest.approx(80.0)

    def test_serialization_roundtrip(self):
        steps = [
            StepReport("step_a", True, 1000, 900, 5.0, {"param": 1}),
            StepReport("step_b", False, 900, 900, 0.1, {}),
        ]
        report = PreprocessingReport(1000, 900, 55.0, steps)
        data = report.to_dict()
        restored = PreprocessingReport.from_dict(data)

        assert restored.input_points == 1000
        assert restored.output_points == 900
        assert restored.total_duration_ms == pytest.approx(55.0)
        assert len(restored.steps) == 2
        assert restored.steps[0].step_name == "step_a"
        assert restored.steps[1].enabled is False

    def test_summary_contains_step_names(self):
        steps = [StepReport("my_filter", True, 500, 450, 3.0)]
        report = PreprocessingReport(500, 450, 10.0, steps)
        summary = report.summary()
        assert "my_filter" in summary
        assert "500" in summary


# ---------------------------------------------------------------------------
# StatisticalOutlierFilter Tests
# ---------------------------------------------------------------------------

class TestStatisticalOutlierFilter:
    def test_removes_outliers(self, noisy_sphere_pcd):
        n_before = len(noisy_sphere_pcd.points)
        f = StatisticalOutlierFilter(nb_neighbors=20, std_ratio=2.0)
        pcd_out, report = f.apply(noisy_sphere_pcd)

        assert len(pcd_out.points) < n_before
        assert report.points_removed > 0
        assert report.enabled is True

    def test_disabled_passes_through(self, noisy_sphere_pcd):
        n_before = len(noisy_sphere_pcd.points)
        f = StatisticalOutlierFilter(enabled=False)
        pcd_out, report = f.apply(noisy_sphere_pcd)

        assert len(pcd_out.points) == n_before
        assert report.enabled is False
        assert report.points_removed == 0

    def test_report_params(self, clean_sphere_pcd):
        f = StatisticalOutlierFilter(nb_neighbors=15, std_ratio=1.5)
        _, report = f.apply(clean_sphere_pcd)
        assert report.params["nb_neighbors"] == 15
        assert report.params["std_ratio"] == pytest.approx(1.5)

    def test_timing_recorded(self, clean_sphere_pcd):
        f = StatisticalOutlierFilter()
        _, report = f.apply(clean_sphere_pcd)
        assert report.duration_ms >= 0.0


# ---------------------------------------------------------------------------
# RadiusOutlierFilter Tests
# ---------------------------------------------------------------------------

class TestRadiusOutlierFilter:
    def test_removes_isolated_points(self, noisy_sphere_pcd):
        n_before = len(noisy_sphere_pcd.points)
        f = RadiusOutlierFilter(search_radius=5.0, min_nb_points=3)
        pcd_out, report = f.apply(noisy_sphere_pcd)
        # Ausreißer liegen weit weg – sollten entfernt werden
        assert len(pcd_out.points) < n_before

    def test_disabled_passes_through(self, clean_sphere_pcd):
        f = RadiusOutlierFilter(enabled=False)
        n_before = len(clean_sphere_pcd.points)
        pcd_out, report = f.apply(clean_sphere_pcd)
        assert len(pcd_out.points) == n_before


# ---------------------------------------------------------------------------
# VoxelGridDownsampler Tests
# ---------------------------------------------------------------------------

class TestVoxelGridDownsampler:
    def test_reduces_points(self, large_pcd):
        n_before = len(large_pcd.points)
        d = VoxelGridDownsampler(voxel_size=2.0)
        pcd_out, report = d.apply(large_pcd)
        assert len(pcd_out.points) < n_before
        assert report.points_removed > 0

    def test_disabled_passes_through(self, large_pcd):
        n_before = len(large_pcd.points)
        d = VoxelGridDownsampler(enabled=False)
        pcd_out, report = d.apply(large_pcd)
        assert len(pcd_out.points) == n_before

    def test_voxel_size_in_params(self, large_pcd):
        d = VoxelGridDownsampler(voxel_size=3.5)
        _, report = d.apply(large_pcd)
        assert report.params["voxel_size"] == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# RandomDownsampler Tests
# ---------------------------------------------------------------------------

class TestRandomDownsampler:
    def test_reduces_to_target(self, large_pcd):
        d = RandomDownsampler(target_points=500)
        pcd_out, _ = d.apply(large_pcd)
        # Nicht exakt, aber nahe dran
        assert len(pcd_out.points) <= 600

    def test_no_upsample(self, clean_sphere_pcd):
        """Wenn Punktwolke bereits kleiner als target_points, keine Änderung."""
        n = len(clean_sphere_pcd.points)
        d = RandomDownsampler(target_points=n + 1000)
        pcd_out, _ = d.apply(clean_sphere_pcd)
        assert len(pcd_out.points) == n


# ---------------------------------------------------------------------------
# NormalEstimator Tests
# ---------------------------------------------------------------------------

class TestNormalEstimator:
    def test_adds_normals(self, clean_sphere_pcd):
        assert not clean_sphere_pcd.has_normals()
        estimator = NormalEstimator(radius=10.0, orient_mode="consistent")
        pcd_out, report = estimator.apply(clean_sphere_pcd)
        assert pcd_out.has_normals()
        assert len(pcd_out.normals) == len(pcd_out.points)

    def test_camera_orientation(self, clean_sphere_pcd):
        estimator = NormalEstimator(
            radius=10.0,
            orient_mode="camera",
            scan_origin=[0.0, 0.0, 200.0],
        )
        pcd_out, _ = estimator.apply(clean_sphere_pcd)
        assert pcd_out.has_normals()

    def test_no_orientation(self, clean_sphere_pcd):
        estimator = NormalEstimator(radius=10.0, orient_mode=None)
        pcd_out, _ = estimator.apply(clean_sphere_pcd)
        assert pcd_out.has_normals()

    def test_disabled_no_normals(self, clean_sphere_pcd):
        estimator = NormalEstimator(enabled=False)
        pcd_out, report = estimator.apply(clean_sphere_pcd)
        assert not pcd_out.has_normals()
        assert report.enabled is False


# ---------------------------------------------------------------------------
# PreprocessingPipeline Tests
# ---------------------------------------------------------------------------

class TestPreprocessingPipeline:
    def test_empty_pipeline(self, clean_sphere_pcd):
        pipeline = PreprocessingPipeline()
        n_before = len(clean_sphere_pcd.points)
        pcd_out, report = pipeline.process(clean_sphere_pcd)
        assert len(pcd_out.points) == n_before
        assert len(report.steps) == 0

    def test_full_pipeline(self, noisy_sphere_pcd):
        pipeline = PreprocessingPipeline([
            StatisticalOutlierFilter(nb_neighbors=20, std_ratio=2.0),
            VoxelGridDownsampler(voxel_size=3.0),
            NormalEstimator(radius=10.0, orient_mode="consistent"),
        ])
        pcd_out, report = pipeline.process(noisy_sphere_pcd)

        assert len(pcd_out.points) < len(noisy_sphere_pcd.points)
        assert pcd_out.has_normals()
        assert len(report.steps) == 3
        assert report.total_duration_ms > 0
        assert report.input_points == len(noisy_sphere_pcd.points)
        assert report.output_points == len(pcd_out.points)

    def test_method_chaining(self, clean_sphere_pcd):
        pipeline = (
            PreprocessingPipeline()
            .add_step(StatisticalOutlierFilter())
            .add_step(VoxelGridDownsampler(voxel_size=5.0))
        )
        assert len(pipeline.steps) == 2

    def test_report_step_order(self, clean_sphere_pcd):
        pipeline = PreprocessingPipeline([
            StatisticalOutlierFilter(),
            VoxelGridDownsampler(voxel_size=5.0),
        ])
        _, report = pipeline.process(clean_sphere_pcd)
        assert report.steps[0].step_name == "statistical_outlier_filter"
        assert report.steps[1].step_name == "voxel_grid_downsampler"

    def test_disabled_step_in_pipeline(self, noisy_sphere_pcd):
        """Deaktivierter Step wird übersprungen aber erscheint im Report."""
        pipeline = PreprocessingPipeline([
            StatisticalOutlierFilter(enabled=False),
            VoxelGridDownsampler(voxel_size=3.0),
        ])
        _, report = pipeline.process(noisy_sphere_pcd)
        assert len(report.steps) == 2
        assert report.steps[0].enabled is False
        assert report.steps[0].points_removed == 0


# ---------------------------------------------------------------------------
# Integration: WeldVolumeModel
# ---------------------------------------------------------------------------

class TestWeldVolumeModelIntegration:
    def test_preprocessing_report_stored(self, noisy_sphere_pcd, tmp_path):
        from schweiss_ki.core.data_structures import WeldVolumeModel

        model = WeldVolumeModel(
            model_id="test_part_01",
            source_type="real",
            source_file=tmp_path / "scan.pcd",
            point_cloud=noisy_sphere_pcd,
        )
        assert not model.has_preprocessing

        pipeline = PreprocessingPipeline([
            StatisticalOutlierFilter(),
            VoxelGridDownsampler(voxel_size=3.0),
        ])
        pcd_clean, report = pipeline.process(model.point_cloud)
        model.update_point_cloud(pcd_clean)
        model.preprocessing_report = report

        assert model.has_preprocessing
        assert model.preprocessing_report.input_points > 0

    def test_save_load_roundtrip(self, noisy_sphere_pcd, tmp_path):
        from schweiss_ki.core.data_structures import WeldVolumeModel

        pipeline = PreprocessingPipeline([
            StatisticalOutlierFilter(),
        ])
        pcd_clean, report = pipeline.process(noisy_sphere_pcd)

        model = WeldVolumeModel(
            model_id="roundtrip_test",
            source_type="real",
            source_file=tmp_path / "scan.pcd",
            point_cloud=pcd_clean,
            preprocessing_report=report,
        )
        model.save(tmp_path)

        loaded = WeldVolumeModel.load(tmp_path / "roundtrip_test")
        assert loaded.has_preprocessing
        assert loaded.preprocessing_report.input_points == report.input_points
        assert loaded.preprocessing_report.output_points == report.output_points
        assert len(loaded.preprocessing_report.steps) == len(report.steps)