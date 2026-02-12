"""
Datenstrukturen für Schweiß-KI AP2.1
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Dict, Any, List
from datetime import datetime
import json

import numpy as np
import open3d as o3d


@dataclass
class WeldVolumeModel:
    """
    3D-Volumenmodell eines Schweißobjekts
    
    Repräsentiert Point Cloud + Metadaten für Ideal/Real/Synthetic Models.
    Dient als zentrale Datenstruktur für Pipeline, Preprocessing und Segmentierung.
    
    Struktur:
    - Identifikation: model_id, source_type, timestamp
    - Geometrie: point_cloud, cad_analysis (optional)
    - Segmentierung: labels, label_names, segmentation_method (später)
    - Metadaten: n_points, density, preprocessing_steps
    """
    
    # === Identifikation ===
    model_id: str
    source_type: Literal["ideal", "real", "synthetic"]
    source_file: Path
    timestamp: datetime = field(default_factory=datetime.now)
    
    # === Geometrie ===
    point_cloud: o3d.geometry.PointCloud = field(default_factory=o3d.geometry.PointCloud)
    cad_analysis: Optional[Dict[str, Any]] = None  # Von Michel's API
    
    # === Segmentierung (später in AP2.1 Phase 3-4) ===
    labels: Optional[np.ndarray] = None
    label_names: Optional[Dict[int, str]] = None
    segmentation_method: Optional[str] = None
    
    # === Metadaten ===
    n_points: int = 0
    density: Optional[float] = None  # Punkte pro mm³
    preprocessing_steps: List[str] = field(default_factory=list)
    
    # Extra Metadaten (flexibel)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Automatische Metadaten-Initialisierung"""
        self.source_file = Path(self.source_file)
        
        # n_points aktualisieren falls Point Cloud vorhanden
        if len(self.point_cloud.points) > 0:
            self.n_points = len(self.point_cloud.points)
            
            # Density schätzen (falls Bounding Box vorhanden)
            if self.density is None:
                bbox = self.point_cloud.get_axis_aligned_bounding_box()
                volume = bbox.volume()
                if volume > 0:
                    self.density = self.n_points / volume  # Punkte pro mm³
    
    @property
    def has_segmentation(self) -> bool:
        """Prüft ob Segmentierung vorhanden"""
        return self.labels is not None
    
    @property
    def has_normals(self) -> bool:
        """Prüft ob Normalen vorhanden"""
        return self.point_cloud.has_normals()
    
    @property
    def has_colors(self) -> bool:
        """Prüft ob Farben vorhanden"""
        return self.point_cloud.has_colors()
    
    def add_preprocessing_step(self, step_name: str):
        """Dokumentiert einen Preprocessing-Schritt"""
        self.preprocessing_steps.append({
            'step': step_name,
            'timestamp': datetime.now().isoformat()
        })
    
    def save(self, output_dir: Path) -> Path:
        """
        Speichert Model in Ordnerstruktur:
        
        output_dir/model_id/
        ├── pointcloud.ply
        └── metadata.json
        """
        output_dir = Path(output_dir)
        model_dir = output_dir / self.model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Point Cloud speichern
        pc_file = model_dir / "pointcloud.ply"
        o3d.io.write_point_cloud(str(pc_file), self.point_cloud)
        
        # Metadata als JSON
        meta_file = model_dir / "metadata.json"
        metadata_dict = {
            # Identifikation
            'model_id': self.model_id,
            'source_type': self.source_type,
            'source_file': str(self.source_file),
            'timestamp': self.timestamp.isoformat(),
            
            # Geometrie
            'cad_analysis': self.cad_analysis,
            
            # Segmentierung
            'has_segmentation': self.has_segmentation,
            'segmentation_method': self.segmentation_method,
            'label_names': self.label_names,
            
            # Metadaten
            'n_points': self.n_points,
            'density': self.density,
            'preprocessing_steps': self.preprocessing_steps,
            'has_normals': self.has_normals,
            'has_colors': self.has_colors,
            
            # Extra
            'metadata': self.metadata,
        }
        
        # Labels speichern (falls vorhanden)
        if self.labels is not None:
            labels_file = model_dir / "labels.npy"
            np.save(labels_file, self.labels)
            metadata_dict['labels_file'] = 'labels.npy'
        
        with open(meta_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        return model_dir
    
    @classmethod
    def load(cls, model_dir: Path) -> 'WeldVolumeModel':
        """
        Lädt Model aus gespeicherter Ordnerstruktur
        """
        model_dir = Path(model_dir)
        
        # Metadata laden
        meta_file = model_dir / "metadata.json"
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        
        # Point Cloud laden
        pc_file = model_dir / "pointcloud.ply"
        point_cloud = o3d.io.read_point_cloud(str(pc_file))
        
        # Labels laden (optional)
        labels = None
        if 'labels_file' in meta:
            labels_file = model_dir / meta['labels_file']
            labels = np.load(labels_file)
        
        # Model rekonstruieren
        model = cls(
            # Identifikation
            model_id=meta['model_id'],
            source_type=meta['source_type'],
            source_file=Path(meta['source_file']),
            timestamp=datetime.fromisoformat(meta['timestamp']),
            
            # Geometrie
            point_cloud=point_cloud,
            cad_analysis=meta.get('cad_analysis'),
            
            # Segmentierung
            labels=labels,
            label_names=meta.get('label_names'),
            segmentation_method=meta.get('segmentation_method'),
            
            # Metadaten
            n_points=meta['n_points'],
            density=meta.get('density'),
            preprocessing_steps=meta.get('preprocessing_steps', []),
            metadata=meta.get('metadata', {}),
        )
        
        return model
    
    def __repr__(self) -> str:
        return (
            f"WeldVolumeModel(id='{self.model_id}', "
            f"type='{self.source_type}', "
            f"points={self.n_points}, "
            f"segmented={self.has_segmentation})"
        )


# Standard Segmentierungs-Labels (für später)
STANDARD_LABELS = {
    0: "background",
    1: "workpiece_left",
    2: "workpiece_right",
    3: "gap_region",
    4: "weld_seam",
}