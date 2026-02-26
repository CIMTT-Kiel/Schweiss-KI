"""
Datenstrukturen für Schweiß-KI AP2.1
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import json
import numpy as np
import open3d as o3d

# Lazily imported to avoid circular imports; type-check only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..preprocessing.base import PreprocessingReport


@dataclass
class WeldVolumeModel:
    """
    3D-Volumenmodell eines Schweißobjekts.

    Repräsentiert Point Cloud + Metadaten für Ideal/Real/Synthetic Models.
    Dient als zentrale Datenstruktur für Pipeline, Preprocessing und Segmentierung.

    Struktur:
    - Identifikation: model_id, source_type, timestamp
    - Geometrie: point_cloud, cad_analysis (optional)
    - Segmentierung: labels, label_names, segmentation_method (Phase 3-4)
    - Metadaten: n_points, density, preprocessing_report
    """

    # === Identifikation ===
    model_id: str
    source_type: Literal["ideal", "real", "synthetic"]
    source_file: Path
    timestamp: datetime = field(default_factory=datetime.now)

    # === Geometrie ===
    point_cloud: o3d.geometry.PointCloud = field(default_factory=o3d.geometry.PointCloud)
    cad_analysis: Optional[Dict[str, Any]] = None  # Von Michel's API

    # === Segmentierung (Phase 3-4) ===
    labels: Optional[np.ndarray] = None
    label_names: Optional[Dict[int, str]] = None
    segmentation_method: Optional[str] = None

    # === Metadaten ===
    n_points: int = 0
    density: Optional[float] = None  # Punkte pro mm³
    preprocessing_report: Optional[Any] = None  # PreprocessingReport (Any für keine Zirkularimporte)

    # Extra Metadaten (flexibel)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Automatische Metadaten-Initialisierung."""
        self.source_file = Path(self.source_file)

        if len(self.point_cloud.points) > 0:
            self.n_points = len(self.point_cloud.points)

            if self.density is None:
                bbox = self.point_cloud.get_axis_aligned_bounding_box()
                volume = bbox.volume()
                if volume > 0:
                    self.density = self.n_points / volume  # Punkte pro mm³

    @property
    def has_segmentation(self) -> bool:
        return self.labels is not None

    @property
    def has_normals(self) -> bool:
        return self.point_cloud.has_normals()

    @property
    def has_colors(self) -> bool:
        return self.point_cloud.has_colors()

    @property
    def has_preprocessing(self) -> bool:
        return self.preprocessing_report is not None

    def update_point_cloud(self, pcd: o3d.geometry.PointCloud) -> None:
        """
        Ersetzt die Punktwolke und aktualisiert n_points und density.
        Typischer Aufruf nach Preprocessing.
        """
        self.point_cloud = pcd
        self.n_points = len(pcd.points)
        bbox = pcd.get_axis_aligned_bounding_box()
        volume = bbox.volume()
        self.density = self.n_points / volume if volume > 0 else None

    def save(self, output_dir: Path) -> Path:
        """
        Speichert Model in Ordnerstruktur:

        output_dir/model_id/
        ├── pointcloud.ply
        ├── labels.npy       (optional)
        └── metadata.json
        """
        output_dir = Path(output_dir)
        model_dir = output_dir / self.model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # Point Cloud
        pc_file = model_dir / "pointcloud.ply"
        o3d.io.write_point_cloud(str(pc_file), self.point_cloud)

        # Metadata
        meta_file = model_dir / "metadata.json"
        metadata_dict = {
            "model_id": self.model_id,
            "source_type": self.source_type,
            "source_file": str(self.source_file),
            "timestamp": self.timestamp.isoformat(),
            "cad_analysis": self.cad_analysis,
            "has_segmentation": self.has_segmentation,
            "segmentation_method": self.segmentation_method,
            "label_names": self.label_names,
            "n_points": self.n_points,
            "density": self.density,
            "has_normals": self.has_normals,
            "has_colors": self.has_colors,
            "preprocessing_report": (
                self.preprocessing_report.to_dict()
                if self.preprocessing_report is not None
                else None
            ),
            "metadata": self.metadata,
        }

        if self.labels is not None:
            labels_file = model_dir / "labels.npy"
            np.save(labels_file, self.labels)
            metadata_dict["labels_file"] = "labels.npy"

        with open(meta_file, "w") as f:
            json.dump(metadata_dict, f, indent=2)

        return model_dir

    @classmethod
    def load(cls, model_dir: Path) -> "WeldVolumeModel":
        """Lädt Model aus gespeicherter Ordnerstruktur."""
        from ..preprocessing.base import PreprocessingReport

        model_dir = Path(model_dir)

        meta_file = model_dir / "metadata.json"
        with open(meta_file, "r") as f:
            meta = json.load(f)

        pc_file = model_dir / "pointcloud.ply"
        point_cloud = o3d.io.read_point_cloud(str(pc_file))

        labels = None
        if "labels_file" in meta:
            labels_file = model_dir / meta["labels_file"]
            labels = np.load(labels_file)

        preprocessing_report = None
        if meta.get("preprocessing_report") is not None:
            preprocessing_report = PreprocessingReport.from_dict(
                meta["preprocessing_report"]
            )

        return cls(
            model_id=meta["model_id"],
            source_type=meta["source_type"],
            source_file=Path(meta["source_file"]),
            timestamp=datetime.fromisoformat(meta["timestamp"]),
            point_cloud=point_cloud,
            cad_analysis=meta.get("cad_analysis"),
            labels=labels,
            label_names=meta.get("label_names"),
            segmentation_method=meta.get("segmentation_method"),
            n_points=meta["n_points"],
            density=meta.get("density"),
            preprocessing_report=preprocessing_report,
            metadata=meta.get("metadata", {}),
        )

    def __repr__(self) -> str:
        return (
            f"WeldVolumeModel(id='{self.model_id}', "
            f"type='{self.source_type}', "
            f"points={self.n_points:,}, "
            f"preprocessed={self.has_preprocessing}, "
            f"segmented={self.has_segmentation})"
        )


# Standard Segmentierungs-Labels (für Phase 3)
STANDARD_LABELS = {
    0: "background",
    1: "workpiece_left",
    2: "workpiece_right",
    3: "gap_region",
    4: "weld_seam",
}