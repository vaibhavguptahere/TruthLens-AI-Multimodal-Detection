# REQUIREMENTS TRACEABILITY MATRIX

**Project**: TruthLens AI — Multimodal NLP for Deepfake Detection and Explainable Content Verification  
**Course**: CSET346 – Natural Language Processing  
**Status Definitions**: `NOT STARTED` | `IN PROGRESS` | `IMPLEMENTED` | `TESTED`

---

| Req ID | Requirement Description | Implementation Location | Validation Method | Status | Notes |
|---|---|---|---|---|---|
| **REQ-01** | Multimodal Input Support (Image + Text Pair) | `src/preprocessing/`, `src/data/` | Unit tests with paired inputs | `IMPLEMENTED` | Curated 450 paired samples ready |
| **REQ-02** | Three-class Output (REAL, FAKE, UNCERTAIN) | `src/models/`, `src/uncertainty/`, `src/data/` | Calibration & confusion matrix | `IMPLEMENTED` | Protocol deterministic & balanced |
| **REQ-03** | Text-only Baseline (DistilBERT/BERT) | `src/models/text_baseline.py` | Standalone evaluation test | `NOT STARTED` | Phase 4 |
| **REQ-04** | Image-only Baseline (ResNet/ViT) | `src/models/image_baseline.py` | Standalone evaluation test | `NOT STARTED` | Phase 4 |
| **REQ-05** | Feature-level Multimodal Fusion | `src/fusion/feature_fusion.py` | Multimodal test script | `NOT STARTED` | Phase 5 |
| **REQ-06** | Decision-level Multimodal Fusion | `src/fusion/decision_fusion.py` | Fusion comparison script | `NOT STARTED` | Phase 5 |
| **REQ-07** | Uncertainty & Confidence Estimation | `src/uncertainty/confidence.py` | Ambiguity threshold checks | `NOT STARTED` | Phase 6 |
| **REQ-08** | Visual Explainability (Grad-CAM) | `src/explainability/gradcam.py` | Visual inspection & heatmap test | `NOT STARTED` | Phase 6 |
| **REQ-09** | Textual Explainability (Token Attribution) | `src/explainability/text_cam.py` | Saliency token score checks | `NOT STARTED` | Phase 6 |
| **REQ-10** | Semantic Consistency Verification | `src/explainability/consistency.py` | Cross-modal cosine tests | `NOT STARTED` | Phase 6 |
| **REQ-11** | Experiment 1: Single vs Multimodal | `experiments/exp1_single_vs_multimodal/`| Metric comparison script | `NOT STARTED` | Phase 7 |
| **REQ-12** | Experiment 2: Fusion Strategy Comparison | `experiments/exp2_fusion_comparison/` | Metric comparison script | `NOT STARTED` | Phase 7 |
| **REQ-13** | Experiment 3: Generalization Evaluation | `experiments/exp3_generalization/` | Out-of-distribution evaluation | `NOT STARTED` | Phase 7 |
| **REQ-14** | Experiment 4: Modality Ablation Study | `experiments/exp4_modality_ablation/` | Ablation performance script | `NOT STARTED` | Phase 7 |
| **REQ-15** | Experiment 5: 15 Difficult-Case Analysis | `experiments/exp5_difficult_cases/` | Qualitative & quantitative log | `NOT STARTED` | Phase 7 |
| **REQ-16** | Experiment 6: Explainability Evaluation | `experiments/exp6_explainability_eval/`| Faithfulness & alignment test | `NOT STARTED` | Phase 7 |
| **REQ-17** | Sanity Checks Suite (8 Specific Edge Cases) | `tests/test_sanity_checks.py` | Pytest test suite | `NOT STARTED` | Phase 7 |
| **REQ-18** | FastAPI Backend Architecture | `backend/api/` | Endpoint integration tests | `NOT STARTED` | Phase 8 |
| **REQ-19** | React Web Interface | `frontend/src/` | End-to-end interactive test | `NOT STARTED` | Phase 8 |
| **REQ-20** | Deliverables & Documentation Suite | `docs/` | Peer review & assignment audit | `IN PROGRESS` | Templates created |
