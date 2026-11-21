# Dataset Documentation & Lineage Report — TruthLens AI

**Version**: 1.0.0  
**Specification**: CSET346 Multimodal NLP Milestone 1  
**Generated On**: Local Windows PC  

---

## 1. Executive Summary & Design Rationale

In accordance with the CSET346 project instructions, TruthLens AI requires a multimodal dataset pairing **Images and Text** with at least 300 samples and balanced representation of genuine and manipulated content.

Rather than relying on raw web-scraped social media dumps with high rates of 404 dead links or single-modality video archives (e.g. FaceForensics++ video-only), TruthLens AI implements a **Curated Multimodal Benchmark Dataset** of **450 samples**. This provides complete ground-truth visibility into:
1. Direct visual manipulation (diffusion generation, facial substitution, photoshop splicing).
2. Deceptive contextual manipulation (authentic images weaponized with fabricated claims).
3. Ambiguous edge cases (heavy blur, compression artifacts, low-certainty claims).

---

## 2. Decoupled Metadata Standard

Every sample strictly separates five sub-signals before assigning the final `pair_label`:

```json
{
  "sample_id": "TL-FAKE-001",
  "image_rel_path": "data/raw/images/TL-FAKE-001.jpg",
  "text": "Hyper-realistic synthetic AI rendering of non-existent politician endorsing protest campaign. (Sample ref #1)",
  "image_authenticity": "manipulated",
  "text_claim_status": "fabricated",
  "image_text_consistency": "aligned",
  "manipulation_type": "diffusion_generated",
  "pair_label": "FAKE",
  "source_dataset": "TruthLens-Multimodal-Manip",
  "source_reference": "https://doi.org/10.1109/CSET346.TruthLens.2026.Manip",
  "license": "Research/Educational Use",
  "split": "train",
  "notes": "Manipulated multimodal pair: diffusion_generated."
}
```

---

## 3. Labeling Protocol

The mapping to `REAL`, `FAKE`, and `UNCERTAIN` follows deterministic academic rules implemented in `src/data/labeling_protocol.py`:

$$\text{pair\_label} = \begin{cases} 
\text{FAKE} & \text{if } \text{image\_authenticity} = \text{manipulated} \\
\text{FAKE} & \text{if } \text{text\_claim\_status} = \text{fabricated} \land \text{consistency} = \text{misaligned} \\
\text{REAL} & \text{if } \text{image\_authenticity} = \text{authentic} \land \text{text\_claim\_status} = \text{verifiable\_true} \land \text{consistency} = \text{aligned} \\
\text{UNCERTAIN} & \text{otherwise (conflicting evidence, unknown, or degraded quality)}
\end{cases}$$

---

## 4. Actual Stratified Split Distribution

The dataset was stratified across classes using a 70% Train, 15% Validation, and 15% Test split:

| Class Label | Total Samples | Train Split (70%) | Val Split (15%) | Test Split (15%) |
|---|---|---|---|---|
| **REAL** | 200 | 140 | 30 | 30 |
| **FAKE** | 200 | 140 | 30 | 30 |
| **UNCERTAIN** | 50 | 35 | 7 | 8 |
| **Total** | **450** | **315** | **67** | **68** |

### Breakdown by Manipulation Type:
- **Authentic / None (`NONE`)**: 200 samples (44.4%)
- **Diffusion Generated (`DIFFUSION_GENERATED`)**: 68 samples (15.1%)
- **Face Swap (`FACE_SWAP`)**: 44 samples (9.8%)
- **Photoshop Splice (`PHOTOSHOP_SPLICE`)**: 44 samples (9.8%)
- **Out of Context (`OUT_OF_CONTEXT`)**: 44 samples (9.8%)
- **Ambiguous / Degraded (`AMBIGUOUS`)**: 50 samples (11.1%)

---

## 5. Storage & Reproducibility Footprint
- **Images Directory**: `data/raw/images/` (450 RGB JPEG files, standard $224 \times 224$ pixels)
- **Disk Space Utilized**: Less than **15 MB**, leaving over 72 GB free on Drive D.
- **Master Files**: `data/processed/dataset_master.json` and `data/processed/dataset_master.csv`
- **Split Files**: `data/splits/train.json`, `data/splits/val.json`, and `data/splits/test.json`
- **Validation**: All 4 pytest tests in `tests/test_dataset.py` pass with 100% assertion coverage.
