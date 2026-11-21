# TruthLens AI: Multimodal NLP for Deepfake Detection & Explainable Content Verification

[![Course](https://img.shields.io/badge/Course-CSET346%20NLP-blue.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU%2FCUDA-orange.svg)](#)
[![Status](https://img.shields.io/badge/Phase-Phase%200%20%26%201%20Completed-brightgreen.svg)](#)

TruthLens AI is a multimodal natural language processing and computer vision framework designed to verify the authenticity of paired text-image content. Unlike single-modality detectors that only scan image pixels or evaluate textual claims, TruthLens AI jointly inspects visual manipulation signatures, textual credibility, and cross-modal semantic consistency to deliver calibrated, explainable verifications: **REAL**, **FAKE**, or **UNCERTAIN**.

---

## 1. Key Features
- **Multimodal Joint Analysis**: Combines language representation (DistilBERT/BERT) with visual representation (ResNet/ViT).
- **Three-Class Decision Protocol**: Explicit handling of ambiguous, conflicting, or low-confidence evidence via **UNCERTAIN**.
- **Dual Explainability**: Visual spatial attention heatmaps (Grad-CAM) + Text saliency attribution + Cross-modal alignment scoring.
- **Comparative Fusion Architecture**: Rigorous bench comparison between Feature-Level Fusion (early/mid) and Decision-Level Fusion (late/ensemble).
- **Local Windows-Compatible Full Stack**: FastAPI REST backend paired with a clean React user interface.

---

## 2. Project Architecture

```
nlp/
├── configs/                # Centralized YAML configurations
├── data/                   # Data storage (git-ignored raw/processed splits)
│   ├── raw/
│   ├── processed/
│   └── splits/
├── src/                    # Core modular Python packages
│   ├── data/               # Dataset ingestion, metadata loaders, schemas
│   ├── preprocessing/      # Image normalization & text tokenization
│   ├── models/             # Image & Text single-modality baselines
│   ├── fusion/             # Feature-level & Decision-level fusion architectures
│   ├── uncertainty/        # Calibration, confidence scoring, uncertainty logic
│   ├── explainability/     # Grad-CAM, text token attribution, consistency analyzer
│   ├── evaluation/         # Evaluation metrics, confusion matrices, plotters
│   └── inference/          # End-to-end multimodal inference pipeline
├── experiments/            # Independent scripts for the 6 required experiments
│   ├── exp1_single_vs_multimodal/
│   ├── exp2_fusion_comparison/
│   ├── exp3_generalization/
│   ├── exp4_modality_ablation/
│   ├── exp5_difficult_cases/
│   └── exp6_explainability_eval/
├── tests/                  # Sanity checks suite (8 edge cases) & unit tests
├── backend/                # FastAPI application
├── frontend/               # React client interface
├── docs/                   # Academic paper, slides, and experiment documentation
├── REQUIREMENTS_TRACEABILITY.md
├── AGENTS.md
└── requirements.txt
```

---

## 3. Local Installation & Setup (Windows PowerShell)

```powershell
# 1. Clone / Navigate to project
cd d:\College_Projects(Academic)\nlp

# 2. (Optional) Create Python Virtual Environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install Dependencies
pip install -r requirements.txt
```

---

## 4. Academic Deliverables & Progress
Refer to [REQUIREMENTS_TRACEABILITY.md](REQUIREMENTS_TRACEABILITY.md) for detailed progress tracking across all CSET346 coursework milestones.
