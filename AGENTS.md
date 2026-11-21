# AGENTS.md — TruthLens AI Project Agent Instructions & Guidelines

## 1. Project Overview
TruthLens AI is an academic research and engineering implementation project for CSET346 (Natural Language Processing).
The core objective is to detect multimodal deepfakes and manipulated content by analyzing paired **Images and Text** and outputting:
- **REAL**
- **FAKE**
- **UNCERTAIN** (with confidence metrics and dual-modality explanations).

## 2. Operating Environment & Constraints
- **OS**: Windows 11 Home (Local PC)
- **Execution Target**: Local machine only (No paid APIs, no Google Colab dependencies)
- **Compute Hardware**: 16 GB System RAM, Intel(R) Iris(R) Xe Graphics (CPU-optimized PyTorch)
- **Pathing**: Always use Python `pathlib.Path` or os-agnostic relative paths. Never hardcode absolute Unix paths.
- **Safety**: Do not delete existing user files or data. Never overwrite files without explicit checks.

## 3. Strict Development Rules
1. **Never fabricate experimental metrics or synthetic evaluation numbers.**
2. **Never claim CLIP cross-modal similarity proves image authenticity.**
3. **Never commit raw dataset images, large archives, or model weight files (.pt, .bin, .safetensors) to Git.**
4. **Always maintain separate metadata for:**
   - `image_authenticity` (manipulated vs authentic image)
   - `text_claim_status` (verifiable vs fabricated caption)
   - `image_text_consistency` (aligned vs misaligned semantic meaning)
   - `manipulation_type`
   - `pair_label` (REAL / FAKE / UNCERTAIN)
   - `source_dataset`
   - `split`
5. **Always update `REQUIREMENTS_TRACEABILITY.md`** when implementing or testing requirements.
