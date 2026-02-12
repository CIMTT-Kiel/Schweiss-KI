# Projektstruktur - Schweiß-KI mit uv & CAD API

## Übersicht
````
schweiss_ki/
├── .python-version          # Python 3.11
├── pyproject.toml           # uv Konfiguration
├── uv.lock                  # Locked dependencies
├── README.md
├── .gitignore
│
├── external/                # External dependencies
│   └── cad-api-client/      # Michel's Client (git submodule oder lokal)
│       ├── client/
│       ├── config.yaml
│       └── notebooks/
│
├── src/
│   └── schweiss_ki/         # Main package (importable)
│       ├── __init__.py
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   └── data_structures.py    # WeldVolumeModel, GapGeometry
│       │
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── cad_api_wrapper.py    # Michel's API Wrapper
│       │   ├── scan_loader.py        # Load PLY/PCD/XYZ
│       │   └── pipeline.py           # End-to-End Orchestration
│       │
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── filtering.py          # Outlier Removal, Denoising
│       │   ├── downsampling.py       # Voxel Grid, Random
│       │   └── normalization.py      # Scaling, Centering
│       │
│       ├── segmentation/
│       │   ├── __init__.py
│       │   ├── ransac.py             # Geometric Segmentation
│       │   ├── clustering.py         # DBSCAN, Region Growing
│       │   └── pointnet/             # ML Segmentation (optional)
│       │       ├── __init__.py
│       │       ├── model.py          # Architecture
│       │       ├── train.py          # Training Loop
│       │       └── inference.py      # Inference
│       │
│       ├── subtraction/              # AP2.2 (später)
│       │   ├── __init__.py
│       │   ├── aligner.py            # ICP Registration
│       │   └── difference.py         # Deviation Computation
│       │
│       ├── quality/                  # AP2.3 (später)
│       │   ├── __init__.py
│       │   ├── metrics.py            # Quality Scores
│       │   └── features.py           # Feature Extraction
│       │
│       └── utils/
│           ├── __init__.py
│           ├── visualization.py      # 3D Plotting
│           ├── io.py                 # File I/O helpers
│           └── metrics.py            # Evaluation Metrics
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # pytest fixtures
│   ├── test_pipeline.py
│   ├── test_cad_api.py               # API Integration Tests
│   ├── test_preprocessing.py
│   ├── test_segmentation.py
│   └── test_data_structures.py
│
├── notebooks/
│   ├── 00_cad_api_exploration.ipynb  # Michel's API testen
│   ├── 01_data_exploration.ipynb
│   ├── 02_ransac_experiments.ipynb
│   ├── 03_pointnet_training.ipynb
│   └── 04_visualization.ipynb
│
├── data/
│   ├── raw/
│   │   ├── step_files/               # Input STEP (CAD)
│   │   │   ├── example_001.step
│   │   │   └── example_002.step
│   │   └── scans/                    # Input 3D Scans (später)
│   │       ├── scan_001.ply
│   │       └── scan_002.pcd
│   │
│   ├── processed/
│   │   ├── ideal/                    # Verarbeitete STEP → Models
│   │   │   ├── model_001_ideal/
│   │   │   │   ├── pointcloud.ply
│   │   │   │   └── metadata.json
│   │   │   └── model_002_ideal/
│   │   │       └── ...
│   │   └── real/                     # Verarbeitete Scans
│   │       ├── model_001_real/
│   │       └── model_002_real/
│   │
│   └── synthetic/                    # Generierte Test-Daten
│       ├── noisy_001.ply
│       └── noisy_002.ply
│
├── docs/
│   ├── AP2.1_plan.md                 # Arbeitsplan
│   ├── architecture.md               # System Architecture
│   └── api.md                        # API Documentation
│
└── scripts/
    ├── setup_env.sh                  # Environment Setup
    ├── setup_cad_api.sh              # CAD API Client Setup
    ├── generate_synthetic.py         # Synthetic Data Generator
    └── batch_process.py              # Batch Processing
````

---

## Setup mit uv & CAD API

### Initial Setup
````bash
# 1. Repository klonen
git clone <repo-url>
cd schweiss_ki

# 2. uv installieren (falls nicht vorhanden)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Python Version festlegen
uv python pin 3.11

# 4. Dependencies installieren
uv sync

# 5. Michel's CAD API Client Setup
git clone <michel-repo-url> external/cad-api-client
cd external/cad-api-client/client
uv sync
cd ../../..

# 6. CAD API Client zu Projekt hinzufügen
uv add --editable external/cad-api-client/client

# 7. Zertifikat von Michel holen und konfigurieren (wenn nötig)
# → In external/cad-api-client/client/config.yaml eintragen
````

### Alternative: Git Submodule (empfohlen)
````bash
# CAD API als Submodule
git submodule add <michel-repo-url> external/cad-api-client
git submodule update --init --recursive

# Setup wie oben
cd external/cad-api-client/client
uv sync
cd ../../..
uv add --editable external/cad-api-client/client
````

---

## Dependencies verwalten
````bash
# Core Dependencies
uv add open3d numpy scipy scikit-learn

# Michel's Client (lokal)
uv add --editable external/cad-api-client/client

# Dev Dependencies
uv add --dev pytest jupyter matplotlib plotly

# ML Dependencies (optional)
uv add --group ml torch pytorch3d

# Synchronisieren
uv sync
````

---

## Tägliche Nutzung
````bash
# Environment aktivieren (automatisch mit uv)
uv run python script.py
uv run pytest
uv run jupyter lab

# Oder manuell aktivieren
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# CAD API testen
python -c "from client.client import CADConverterClient; print('✓ API Client verfügbar')"
````

---

## pyproject.toml
````toml
[project]
name = "schweiss-ki"
version = "0.1.0"
description = "AI-gestützte Schweißnaht-Qualitätskontrolle"
authors = [
    {name = "Fabian Heinze", email = "fabian.heinze@haw-kiel.de"}
]
readme = "README.md"
requires-python = ">=3.11"

dependencies = [
    "open3d>=0.18.0",
    "numpy>=1.24.0",
    "scipy>=1.11.0",
    "scikit-learn>=1.3.0",
    # Michel's CAD API Client (lokal)
    "cad-api-client @ file:///external/cad-api-client/client",
]

[project.optional-dependencies]
ml = [
    "torch>=2.0.0",
    "pytorch3d>=0.7.5",
]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "jupyter>=1.0.0",
    "matplotlib>=3.7.0",
    "plotly>=5.14.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I"]
ignore = []
````

---

## CAD API Wrapper
````python
# src/schweiss_ki/pipeline/cad_api_wrapper.py

from pathlib import Path
from client import CADConverterClient
import open3d as o3d

class CADConverter:
    """Wrapper für Michel's CAD API"""
    
    def __init__(self, config_file: Path = None):
        self.client = CADConverterClient(config_file=config_file)
    
    def step_to_pointcloud(
        self, 
        step_file: Path, 
        output_file: Path = None
    ) -> o3d.geometry.PointCloud:
        """
        STEP → Point Cloud direkt
        
        Args:
            step_file: Input STEP file
            output_file: Output PLY (optional)
        
        Returns:
            Open3D PointCloud
        """
        if output_file is None:
            output_file = step_file.with_suffix('.ply')
        
        # API Call: STEP → PLY
        ply_file = self.client.convert_to_ply(
            str(step_file), 
            str(output_file)
        )
        
        # Load mit Open3D
        pcd = o3d.io.read_point_cloud(str(ply_file))
        
        return pcd
    
    def analyse_cad(self, step_file: Path) -> dict:
        """CAD-Geometrie analysieren (optional)"""
        return self.client.analyse_cad(str(step_file))
````

---

## Beispiel: Pipeline mit CAD API
````python
# src/schweiss_ki/pipeline/pipeline.py

from pathlib import Path
from .cad_api_wrapper import CADConverter
from ..core.data_structures import WeldVolumeModel
from ..preprocessing import Preprocessor
from ..segmentation import RANSACSegmenter

class Pipeline:
    def __init__(self):
        self.cad_converter = CADConverter()
        self.preprocessor = Preprocessor()
        self.segmenter = RANSACSegmenter()
    
    def process_step(self, step_file: Path) -> WeldVolumeModel:
        """
        End-to-End: STEP → WeldVolumeModel
        """
        # 1. STEP → Point Cloud (via Michel's API)
        pcd = self.cad_converter.step_to_pointcloud(step_file)
        
        # 2. Preprocessing
        pcd_clean = self.preprocessor.process(pcd)
        
        # 3. Segmentation
        labels = self.segmenter.segment(pcd_clean)
        
        # 4. WeldVolumeModel erstellen
        model = WeldVolumeModel(
            model_id=step_file.stem,
            source_type="ideal",
            source_file=step_file,
            point_cloud=pcd_clean,
            labels=labels
        )
        
        return model
````

---

## .gitignore
````
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Virtual Environment
.venv/
venv/
ENV/

# uv
.python-version
uv.lock

# IDE
.vscode/
.idea/
*.swp
*.swo

# External (CAD API)
external/cad-api-client/client/config.yaml  # Zertifikat nicht committen!
external/cad-api-client/client/.venv/

# Data (große Files nicht committen)
data/raw/step_files/*.step
data/raw/scans/*.ply
data/raw/scans/*.pcd
data/processed/*/pointcloud.ply
data/synthetic/*.ply

# Aber: Git-Struktur behalten
!data/raw/step_files/.gitkeep
!data/raw/scans/.gitkeep

# Notebooks
.ipynb_checkpoints/
notebooks/*.nbconvert.ipynb

# Testing
.pytest_cache/
.coverage
htmlcov/

# OS
.DS_Store
Thumbs.db
````

---

## README.md (Beispiel)
````markdown
# Schweiß-KI - AP2.1

KI-gestützte Schweißnaht-Qualitätskontrolle

## Quick Start
```bash
# Setup
git submodule update --init --recursive
cd external/cad-api-client/client && uv sync && cd ../../..
uv sync

# Zertifikat konfigurieren
# → Siehe external/cad-api-client/client/config.yaml

# Test Pipeline
source .venv/bin/activate
python -m schweiss_ki.pipeline.pipeline \
    --input data/raw/step_files/example.step \
    --output data/processed/example.ply

# Tests
uv run pytest

# Notebooks
uv run jupyter lab
```

## Dependencies

**Core:**
- open3d, numpy, scipy, scikit-learn
- Michel's CAD API Client (via git submodule)

**ML (optional):**
- torch, pytorch3d

**Dev:**
- pytest, jupyter, matplotlib

## CAD API Setup

Michel's CAD Conversion API läuft als Service.

1. Client-Repo als Submodule
2. Zertifikat von Michel holen
3. In `external/cad-api-client/client/config.yaml` eintragen
4. Test: `python -c "from client import CADConverterClient; print('OK')"`

Details: Siehe `docs/cad_api_setup.md`

## Project Structure

- `src/schweiss_ki/` - Main package
- `external/cad-api-client/` - Michel's API Client (submodule)
- `tests/` - Unit tests
- `notebooks/` - Jupyter notebooks
- `data/` - Data storage (gitignored)
````

---

## Deployment auf anderen Rechnern

### Heidenbluth/HAW Installation
````bash
# 1. Repository klonen (mit Submodules)
git clone --recursive <repo-url>
cd schweiss_ki

# 2. uv installieren
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. CAD API Client Setup
cd external/cad-api-client/client
uv sync
# Zertifikat eintragen in config.yaml
cd ../../..

# 4. Main Project Setup
uv sync

# 5. Test
uv run python -c "from client import CADConverterClient; print('✓')"
uv run python -m schweiss_ki.pipeline.pipeline --help
````

**Vorteile:**
- ✓ Alles via Git tracked (Submodule)
- ✓ Reproduzierbar durch uv.lock
- ✓ Keine manuelle FreeCAD Installation
- ✓ API-basiert → deployment-freundlich

---

## Scripts

### Setup Script (scripts/setup_cad_api.sh)
````bash
#!/bin/bash
# Setup CAD API Client

echo "Setting up CAD API Client..."

# Check if submodule exists
if [ ! -d "external/cad-api-client" ]; then
    echo "Error: CAD API submodule not found"
    echo "Run: git submodule update --init --recursive"
    exit 1
fi

# Setup Client
cd external/cad-api-client/client
uv sync

# Check config
if [ ! -f "config.yaml" ]; then
    echo "Warning: config.yaml not found"
    echo "Please obtain certificate from Michel and configure"
fi

cd ../../..
echo "✓ CAD API Client setup complete"
````

---

## Performance

**Mit Michel's API (erwartete Zeiten):**

- STEP → PLY (API): 2-5 Sekunden (Server-abhängig)
- PLY laden (Open3D): < 1 Sekunde
- Preprocessing: 0.5-1 Sekunde
- RANSAC Segmentation: 2-5 Sekunden
- PointNet Inference: 0.1-0.5 Sekunden (GPU)

**Total: ~5-15 Sekunden pro Model** (inkl. API-Call)

---

**Letzte Aktualisierung:** 2025-02-09