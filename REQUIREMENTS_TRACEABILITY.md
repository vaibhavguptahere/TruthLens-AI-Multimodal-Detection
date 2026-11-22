# REQUIREMENTS TRACEABILITY MATRIX

**Project**: TruthLens AI — Multimodal NLP for Deepfake Detection and Explainable Content Verification  
**Course**: CSET346 – Natural Language Processing  
**Status Definitions**: `NOT STARTED` | `IN PROGRESS` | `IMPLEMENTED` | `TESTED`

---

| Req ID | Requirement Description | Implementation Location | Validation Method | Status | Notes |
|---|---|---|---|---|---|
| **REQ-01** | Multimodal Input Support (Image + Text Pair) | `src/preprocessing/`, `src/data/` | Unit tests with paired inputs | `TESTED` | Fully tested in test_preprocessing.py |
| **REQ-02** | Three-class Output (REAL, FAKE, UNCERTAIN) | `src/models/`, `src/uncertainty/`, `src/data/` | Calibration & confusion matrix | `IMPLEMENTED` | Protocol deterministic & balanced |
| **REQ-03** | Text-only Baseline (DistilBERT/BERT) | `src/models/text_baseline.py` | Standalone evaluation test | `TESTED` | Forward pass & proba verified in test_baselines.py |
| **REQ-04** | Image-only Baseline (ResNet/ViT) | `src/models/image_baseline.py` | Standalone evaluation test | `TESTED` | ResNet-50 head verified in test_baselines.py |
| **REQ-05** | Feature-level Multimodal Fusion | `src/fusion/feature_fusion.py` | Multimodal test script | `TESTED` | Forward pass & embeddings verified in test_fusion.py |
| **REQ-06** | Decision-level Multimodal Fusion | `src/fusion/decision_fusion.py` | Fusion comparison script | `TESTED` | Conflict resolution & UNCERTAIN routing verified in test_fusion.py |
| **REQ-07** | Uncertainty & Confidence Estimation | `src/uncertainty/confidence.py`, `src/fusion/decision_fusion.py` | Ambiguity threshold checks | `TESTED` | Calibrated status & conflict routing verified in test_fusion.py & test_explainability.py |
| **REQ-08** | Visual Explainability (Grad-CAM) | `src/explainability/gradcam.py` | Visual inspection & heatmap test | `TESTED` | Normalized 224x224 heatmap verified in test_explainability.py |
| **REQ-09** | Textual Explainability (Token Attribution) | `src/explainability/text_saliency.py` | Saliency token score checks | `TESTED` | Gradient norm token attribution verified in test_explainability.py |
| **REQ-10** | Semantic Consistency Verification | `src/explainability/consistency.py` | Cross-modal cosine tests | `TESTED` | Cosine similarity & mismatch detection verified in test_explainability.py |
| **REQ-11** | Experiment 1: Single vs Multimodal | `experiments/exp1_single_vs_multimodal/`| Metric comparison script | `TESTED` | Evaluated & saved to exp1_metrics.csv |
| **REQ-12** | Experiment 2: Fusion Strategy Comparison | `experiments/exp2_fusion_comparison/` | Metric comparison script | `TESTED` | Evaluated & saved to exp2_metrics.csv |
| **REQ-13** | Experiment 3: Generalization Evaluation | `experiments/exp3_generalization/` | Out-of-distribution evaluation | `TESTED` | Evaluated across 6 manipulation types |
| **REQ-14** | Experiment 4: Modality Ablation Study | `experiments/exp4_modality_ablation/` | Ablation performance script | `TESTED` | Evaluated & saved to exp4_ablation_metrics.csv |
| **REQ-15** | Experiment 5: 15 Difficult-Case Analysis | `experiments/exp5_difficult_cases/` | Qualitative & quantitative log | `TESTED` | 15 detailed cases saved to difficult_cases_analysis.json |
| **REQ-16** | Experiment 6: Explainability Evaluation | `experiments/exp6_explainability_eval/`| Faithfulness & alignment test | `TESTED` | Grad-CAM & Saliency ratios saved to explainability_eval.csv |
| **REQ-17** | Sanity Checks Suite (8 Specific Edge Cases) | `tests/test_sanity_checks.py` | Pytest test suite | `TESTED` | 8/8 test scenarios passed |
| **REQ-18** | FastAPI Backend Architecture | `backend/api/` | Endpoint integration tests | `TESTED` | /api/verify & /api/health passed in test_api.py |
| **REQ-19** | React Web Interface | `frontend/index.html` | End-to-end interactive test | `TESTED` | Interactive Tailwind + React SPA verified |
| **REQ-20** | Deliverables & Documentation Suite | `docs/` | Peer review & assignment audit | `TESTED` | Complete IEEE paper, slides & docs compiled |
