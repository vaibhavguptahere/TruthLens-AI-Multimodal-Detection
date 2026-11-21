# Experimental Methodology & Protocols

## 1. Overview
This document specifies the protocols for the six required experiments in TruthLens AI.

## 2. Experiments
- **Experiment 1**: Single-Modality vs Multimodal Performance Comparison.
- **Experiment 2**: Fusion Strategy Benchmark (Feature-Level vs Decision-Level).
- **Experiment 3**: Generalization to Unseen Manipulation Types or Out-of-Domain Data.
- **Experiment 4**: Modality Ablation (Image-only vs Text-only vs Multimodal).
- **Experiment 5**: Qualitative & Quantitative Analysis of 15 Difficult / Corner Cases.
- **Experiment 6**: Explainability Evaluation (Grad-CAM localization relevance & Token attribution faithfulness).

## 3. Metrics Protocol
All experiments report:
- Accuracy, Macro/Weighted Precision, Recall, F1-Score
- Multi-class ROC-AUC / PR-AUC
- Confusion Matrices (saved as high-res PNG and normalized CSV)
