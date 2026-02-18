# Schweiß-KI - Welding Quality Control via 3D Volume Modeling

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![uv](https://img.shields.io/badge/package_manager-uv-blue.svg)](https://github.com/astral-sh/uv)
[![Status](https://img.shields.io/badge/status-AP2.1_Phase_1-yellow.svg)]()

AI-assisted 3D quality control system for automated welding processes. Developed by the Center for Industrial Manufacturing Technology and Transfer (CIMTT) at Kiel University of Applied Sciences in cooperation with Heidenbluth GmbH.

## 🎯 Project Overview

This repository implements a 3D volume model for welding quality assessment, focusing on:

- **Geometry Deviation Detection**: Identify deviations in weld seams (±0.25 mm tolerance) across various joint geometries
- **CAD to Point Cloud Pipeline**: Convert STEP files to processed point clouds via API
- **Point Cloud Segmentation**: Segment weld seam regions using geometric (RANSAC) and ML (PointNet) approaches
- **Quality Assessment**: Digital twin-based quality scoring for reinforcement learning optimization

**Current Focus:** AP2.1 - Data flow, preprocessing & segmentation pipeline

The project addresses thermally induced distortions in multi-pass welding by developing robust 3D scanning and quality assessment methods.

## 🏗️ Project Structure
```
schweiss-ki/
├── README.md                        # This file
├── pyproject.toml                   # Project configuration and dependencies
├── uv.lock                          # Locked dependencies for reproducibility

├── external/                        # External dependencies
│   └── cad-api-client/              # CAD conversion API (submodule)

├── src/
│   └── schweiss_ki/                 # Main package
│       ├── core/
│       │   └── data_structures.py   # WeldVolumeModel, GapGeometry
│       ├── pipeline/
│       │   ├── cad_api_wrapper.py   # 🚧 CAD API integration
│       │   ├── scan_loader.py       # 📋 Point cloud loaders (PLY/PCD/XYZ)
│       │   └── pipeline.py          # 📋 End-to-end orchestration
│       ├── preprocessing/
│       │   ├── filtering.py         # 📋 Outlier removal, denoising
│       │   ├── downsampling.py      # 📋 Voxel grid, adaptive methods
│       │   └── normalization.py     # 📋 Scaling, centering
│       ├── segmentation/
│       │   ├── ransac.py            # 📋 Geometric segmentation
│       │   ├── clustering.py        # 📋 DBSCAN, region growing
│       │   └── pointnet/            # 📋 ML-based segmentation (optional)
│       ├── subtraction/             # 📋 AP2.2 - CAD vs. Real deviation
│       ├── quality/                 # 📋 AP2.3 - Quality metrics & digital twin
│       └── utils/
│           ├── visualization.py     # 📋 3D plotting
│           └── io.py                # 📋 File I/O helpers

├── scripts/
│   ├── test_cad_conversion.py       # ✅ CAD API test script
│   └── batch_process.py             # 📋 Batch processing utilities

├── tests/                           # 📋 Unit tests
├── notebooks/                       # 📋 Jupyter notebooks for exploration
├── data/                            # Data storage (gitignored)
│   ├── raw/                         # Input STEP files & scans
│   ├── processed/                   # Processed point clouds
│   └── synthetic/                   # Generated test data
└── docs/                            # Project documentation and planning

```

**Legend:** ✅ Implemented | 🚧 In Progress | 📋 Planned

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- Access to CAD API (certificate required)
- Optional: CUDA-capable GPU for ML segmentation

### Installation

1. **Clone the repository (with submodules)**
```bash
   git clone --recursive git@github.com:CIMTT-Kiel/schweiss-ki.git
   cd schweiss-ki
```

2. **Install using uv (recommended)**
```bash
   # Install uv if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Install project dependencies
   uv sync
```

3. **Alternative: Install with pip**
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .
```

4. **Configure CAD API access**
```bash
   # Obtain certificate from API provider
   # Place in: external/cad-api-client/config.yaml
```

### Quick Test
```bash
# Test CAD API connection
uv run python -c "from client.core import CADConverterClient; c = CADConverterClient(); print(c.get_service_status())"

# Run STEP → Point Cloud conversion test
uv run python scripts/test_cad_conversion.py
```

Expected output:
```
✅ Konvertierung erfolgreich!
   Dauer: 0.28 Sekunden
   Punkte: 8,192
```

## 🔧 Work Packages (Arbeitspakete)

### AP2.1 - Data Flow, Preprocessing & Segmentation (2.5 PM)
**Status:** Phase 1 nearly complete, Phase 2 starting

**Completed:**
- ✅ CAD conversion pipeline (STEP → Point Cloud via API)
- ✅ `WeldVolumeModel` data structure implemented
- ✅ Batch processing tested with sample STEP files

**In Progress:**
- 🚧 Integration into automated pipeline
- 🚧 Preprocessing module (statistical outlier filtering, downsampling)

**Planned:**
- 📋 RANSAC-based segmentation (Week 6-10)
- 📋 PointNet segmentation (optional, data-dependent)

**Critical Dependencies:**
- ⚠️ Real 3D scan data from Heidenbluth (required by Week 4)

### AP2.2 - Subtraction Method (2.0 PM)
**Status:** Planned (starts after AP2.1 milestone)

- 3D registration (ICP alignment)
- Deviation analysis (CAD ideal vs. real scan)
- Tolerance classification (±0.25 mm)

### AP2.3 - Digital Twin & Quality Assessment (2.0 PM)
**Status:** Planned

- Digital twin modeling
- Automated quality scoring
- Feature extraction for RL optimization (AP3)

### AP2.4 - Volume Calculation (2.5 PM)
**Status:** Future work

### AP2.5 - Real-time Data Compression (1.75 PM)
**Status:** Future work

## 📊 Data Structure

### WeldVolumeModel
Central data structure for all work packages:
```python
WeldVolumeModel:
├── Identification
│   ├── model_id: str
│   ├── source_type: "ideal" | "real" | "synthetic"
│   └── timestamp: datetime
├── Geometry
│   ├── point_cloud: PointCloud (Open3D)
│   └── cad_metadata: dict (optional)
├── Segmentation
│   ├── labels: array[N]
│   ├── label_names: dict
│   └── method: "ransac" | "pointnet" | "hybrid"
└── Metadata
    ├── n_points: int
    └── preprocessing_steps: list
```

### Segmentation Labels
- `0`: Background
- `1`: Workpiece Flank A
- `2`: Workpiece Flank B
- `3`: Gap Region (weld seam area)

## 🧠 Planned Algorithms

### 1. Geometric Segmentation (RANSAC)
- **Purpose**: Robust plane detection for V-groove identification
- **Input**: Preprocessed point cloud
- **Output**: Segmented regions (flanks, gap, background)
- **Advantage**: No training data required

### 2. ML-based Segmentation (PointNet)
- **Purpose**: Learning-based segmentation refinement
- **Input**: Point cloud + optional features
- **Output**: Point-wise segmentation labels
- **Advantage**: Handles complex geometries, adaptable
- **Requirement**: Labeled training data

### 3. Hybrid Approach
- **Purpose**: Combine geometric robustness with ML accuracy
- **Method**: RANSAC for coarse segmentation → PointNet for refinement

## 📈 Evaluation Metrics

**Preprocessing:**
- Point retention rate (%)
- Noise reduction (σ)
- Processing time (s/100k points)

**Segmentation:**
- Segmentation accuracy (target: >90% for synthetic, >80% for real data)
- Per-class precision and recall
- Visual quality assessment

**Pipeline Performance:**
- Throughput (models/hour)
- Memory usage (GB)
- End-to-end latency (seconds)

## 🛠️ Development

### Dependencies

**Core:**
- `open3d` - Point cloud processing
- `numpy`, `scipy`, `scikit-learn` - Numerical computing
- `client` (submodule) - CAD API integration

**Optional (ML):**
- `torch`, `pytorch3d` - Deep learning

**Dev:**
- `pytest`, `jupyter`, `matplotlib`

### Running Tests
```bash
uv run pytest tests/
```

### Code Style

The project follows standard Python conventions:
- Line length: 100 characters
- Type hints encouraged
- Docstrings: Google style

## 📚 Documentation

- **[Project Plan AP2](docs/20260217_Projektplan_AP2_FH_V1_0.md)**: Detailed work package planning
- **[Technical Architecture](docs/AP2.1_structure.md)**: System design and module structure
- **[CAD API Docs](external/cad-api-client/)**: API integration documentation

## 👥 Team

**Project Lead**: Fabian Heinze  
Email: fabian.heinze@haw-kiel.de  
Organization: CIMTT, Kiel University of Applied Sciences

**Partners:**
- Heidenbluth GmbH (Industry Partner - Welding Equipment)
- Michel Kruse (CAD API Integration)

**Supervisor**: Prof. Dr.-Ing. Alexander Mattes
HAW Kiel, Department of Mechanical Engineering

## 🔗 Related Projects

- **[CAD Preprocessing API](https://github.com/CIMTT-Kiel/cad-api-client)**: STEP to Point Cloud conversion service


## 📄 License & Confidentiality

🔒 **Confidential Research Project** (ZIM Cooperation)  
Not for distribution to third parties without permission.

Developed under ZIM funding program (BMWi).

## 📞 Support and Contact

- **Issues**: Report via GitHub Issues (internal team only)
- **Email**: fabian.heinze@haw-kiel.de
- **Organization**: [CIMTT Website](https://www.fh-kiel.de/cimtt)

## 📚 Citation

If you reference this work in academic publications:
```bibtex
@software{schweiss_ki_2026,
  title = {Schweiß-KI: AI-Assisted 3D Quality Control for Automated Welding},
  author = {Fabian Heinze},
  organization = {CIMTT, Kiel University of Applied Sciences},
  year = {2026},
  url = {https://github.com/CIMTT-Kiel/schweiss-ki},
  note = {ZIM Cooperation Project with Heidenbluth GmbH}
}
```

---

**Developed by CIMTT at Kiel University of Applied Sciences**  
**Funded by:** ZIM (Zentrales Innovationsprogramm Mittelstand)