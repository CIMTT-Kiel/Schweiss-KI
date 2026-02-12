# AP2.1 Arbeitsplan - Optimierter Datenfluss

**Projekt:** Schweiß-KI  
**Arbeitspaket:** AP2.1 (HAW Kiel, 2.5 Personenmonate)  
**Zeitraum:** 2-3 Monate  

---

## Ziel

Entwicklung eines optimierten Datenflusses zur Aufbereitung und Segmentierung von Punktwolken mit RANSAC und PointNet, um präzise 3D-Volumenmodelle mit variierenden Geometrien zu erstellen (±0.25mm Toleranz).

---

## Aufteilung

### AP2.1a: Foundation - Synthetische Daten (6-8 Wochen)
**Input:** STEP Files, Synthetic Scans  
**Output:** Funktionierende Pipeline, Basis-Algorithmen

### AP2.1b: Real Data Integration (4-6 Wochen, später)
**Input:** Echte 3D Scans von Heidenbluth  
**Output:** Robuste Production-Pipeline

---

## AP2.1a: Foundation (Synthetische Daten)

### Phase 1: Pipeline Setup (2 Wochen)

**Ziel:** STEP → Segmentierte Point Cloud

**Tasks:**
- Michel's CAD API Setup (Client-Repo, Zertifikat)
- STEP → PLY direkt (API-Call)
- PLY laden mit Open3D
- Synthetic Scan Generator (Gaussian Noise 0.05-0.1mm)
- WeldVolumeModel mit Segmentierungs-Support

**API Integration:**
````python
from client.client import CADConverterClient

client = CADConverterClient()
ply_file = client.convert_to_ply("model.step", "output.ply")
pcd = o3d.io.read_point_cloud(str(ply_file))
````

**Deliverable:**
- 10+ Test-Models (Ideal + Synthetic)
- Pipeline funktioniert mit Michel's API

---

### Phase 2: Preprocessing (2 Wochen)

**Ziel:** Robuste Point Cloud Aufbereitung

**Tasks:**
- Statistical Outlier Removal (KNN, k=20)
- Voxel Grid Downsampling (adaptive)
- Normal Estimation & Orientation
- ROI Extraction

**Deliverable:**
- Preprocessing-Module konfigurierbar
- Vor/Nach Vergleiche

**Metriken:**
- Point Retention Rate
- Normal Consistency
- Processing Time

---

### Phase 3: RANSAC Segmentierung (2 Wochen)

**Ziel:** Geometric Segmentation

**Tasks:**
- Plane Detection (RANSAC, threshold 0.25mm)
- Line/Edge Detection (Gap-Kanten)
- Clustering (DBSCAN, eps=0.5mm)
- Label Assignment (Background, Workpiece L/R, Gap)

**Deliverable:**
- Segmentierung mit IoU > 0.9
- Labels in WeldVolumeModel

**Algorithmus:**
````
1. Detect Planes (2x Workpiece) → RANSAC
2. Remove Inliers → Gap + Background bleiben
3. Cluster Gap Region → DBSCAN
4. Assign Labels → 5 Klassen
````

---

### Phase 4: PointNet (2 Wochen, Optional)

**Ziel:** ML-basierte Segmentierung

**Tasks:**
- PointNet Architecture (Vanilla)
- Training Data (100+ Samples, Augmentation)
- Training (Cross-Entropy, Adam, 50-100 Epochs)
- Hybrid: RANSAC + PointNet Fusion

**Deliverable:**
- Trainiertes Model
- IoU Vergleich: RANSAC vs. PointNet vs. Hybrid

**Status:** Optional - wenn Zeit, sonst später

---

### Milestone AP2.1a
````
✓ Pipeline: STEP → Segmented Point Cloud (via API)
✓ Preprocessing: Robust gegen Noise
✓ Segmentation: IoU > 0.9
✓ 20+ Test-Models
✓ Code dokumentiert & getestet
````

---

## AP2.1b: Real Data Integration

### Vorbedingungen

**Von Heidenbluth benötigt:**
- STEP Files (CAD-Modelle)
- 3D Scans (Format klären: PLY/PCD/XYZ?)
- Metadaten (Sensor, Auflösung)
- Optional: Ground Truth Labels

---

### Phase 1: Real Data Pipeline (2 Wochen)

**Tasks:**
- Scan Loader (multi-format)
- Data Exploration & Analyse
- Format-Konvertierung zu PLY
- Koordinatensystem-Anpassung

**Deliverable:**
- Real Scans laden in WeldVolumeModel
- Dokumentation: Scan-Eigenschaften

**Erwartete Probleme:**
- Incomplete Coverage (Occlusion)
- Höheres Noise
- Outliers (Spatter, Reflections)
- Ungleichmäßige Density

---

### Phase 2: Preprocessing Anpassung (2 Wochen)

**Tasks:**
- Aggressive Outlier Removal
- Adaptive Downsampling (hohe Density an Gap)
- Missing Data Detection
- Adaptive Filter-Parameter

**Deliverable:**
- Preprocessing für Real Scans
- Parameter-Tuning dokumentiert

---

### Phase 3: Segmentation Robustness (2 Wochen)

**Tasks:**
- RANSAC Parameter-Tuning (höhere Toleranzen)
- Outlier-resistentes Clustering
- PointNet Fine-Tuning (falls vorhanden)
- Hybrid Approach Refinement

**Deliverable:**
- Segmentierung: IoU > 0.8 für Real Data
- Failure Cases dokumentiert

**Validation:**
- Manuelle Labels (Subset)
- Visual Inspection
- Vergleich Ideal vs. Real

---

### Milestone AP2.1b
````
✓ Pipeline: Real Scans → Segmented Point Cloud
✓ Preprocessing: Robust gegen Real-World Noise
✓ Segmentation: IoU > 0.8
✓ Ready für AP2.2/2.3/2.4
````

---

## Datenstruktur
````python
WeldVolumeModel:
├── Identifikation
│   ├── model_id: str
│   ├── source_type: "ideal" | "real" | "synthetic"
│   └── timestamp: datetime
│
├── Geometrie
│   ├── point_cloud: PointCloud
│   └── cad_analysis: dict (optional, von API)
│
├── Segmentierung
│   ├── labels: array[N]
│   ├── label_names: dict
│   └── segmentation_method: str
│
└── Metadaten
    ├── n_points: int
    ├── density: float
    └── preprocessing_steps: list

Segmentierungs-Labels:
- 0: Background
- 1: Workpiece Left
- 2: Workpiece Right
- 3: Gap Region
- 4: Weld Seam (optional)
````

---

## Tech Stack

**Core:**
- `open3d` - Point Cloud Processing
- `numpy` - Numerics
- `scipy` - Algorithms
- **Michel's CAD API Client** - STEP → PLY

**Segmentation:**
- `scikit-learn` - DBSCAN, Metrics

**ML (Optional):**
- `torch` - PointNet
- `pytorch3d` - 3D Operations

**Dev:**
- `pytest` - Testing
- `jupyter` - Exploration
- `matplotlib` - Visualization

---

## Setup

### CAD API Client
````bash
# Michel's Client-Repo klonen
git clone <michel-repo-url>
cd client
uv sync

# Zertifikat von Michel holen
# In config.yaml eintragen

# Test
python -c "from client import CADConverterClient; print('OK')"
````

### Eigenes Projekt
````bash
cd schweiss_ki
uv add ../path/to/michel-client  # Lokaler Pfad
# ODER wenn Michel's Client als Package:
# uv add git+<michel-repo-url>
````

---

## Evaluation Metrics

**Preprocessing:**
- Point Retention (%)
- Noise Reduction (σ)
- Processing Time (s/100k points)

**Segmentation:**
- IoU per class
- Overall Accuracy (%)
- Precision/Recall
- Confusion Matrix

**Pipeline:**
- Throughput (models/hour)
- Memory Usage (GB)
- Latency (seconds)

---

## Success Criteria

**AP2.1a:**
````
✓ Pipeline automatisiert (mit Michel's API)
✓ IoU > 0.9 (synthetic)
✓ 20+ Test-Models
✓ Code clean & dokumentiert
````

**AP2.1b:**
````
✓ Real Scans verarbeitbar
✓ IoU > 0.8 (real)
✓ Robust gegen Noise
✓ Output für AP2.2/2.3 ready
````

---

## Schnittstellen

**→ AP2.2 (Subtraktionsmethode):**
- WeldVolumeModel (Ideal + Real, aligned, segmentiert)

**→ AP2.3 (Quality Assessment):**
- Segmentierte Regions für region-specific Metrics

**→ AP2.4 (Volume Calculation):**
- Gap Segmentation für Volumen-Berechnung

**→ Michel's CAD API:**
- Gemeinsame Weiterentwicklung STEP-Import
- Optional: `analyse_cad()` für Metadaten

---

## Vorteile Michel's API

- ✓ Keine lokale FreeCAD/OpenCASCADE Installation
- ✓ Deployment-freundlich (API auf Server)
- ✓ Direkt STEP → PLY (1 Schritt weniger)
- ✓ Robuster (zentral maintained)
- ✓ Bonus Features: CAD-Analyse, VecSet, Voxel, etc.

---

## Nächste Schritte

**Woche 1:**
1. Michel's Client-Repo Setup + Zertifikat
2. Notebook durcharbeiten (Beispiele)
3. Eigene STEP Files testen
4. Integration in `pipeline/step_converter.py`

**Woche 2:**
5. WeldVolumeModel implementieren
6. Synthetic Scan Generator
7. 10 Test-Models via API

**Review nach 2 Wochen:**
- Pipeline funktioniert mit API?
- PLY-Qualität ausreichend?
- Weiter zu Preprocessing!