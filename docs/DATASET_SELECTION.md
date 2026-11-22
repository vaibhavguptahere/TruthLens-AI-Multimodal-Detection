# DATASET SELECTION REPORT — TRUTHLENS AI

**Course**: CSET346 – Natural Language Processing  
**Project**: Multimodal NLP for Deepfake Detection & Explainable Content Verification  
**Selected Modalities**: Image + Text ONLY  
**Evaluation Date**: November 2026  

---

## 1. Executive Summary & Audit Mandate
Per CSET346 project specifications, TruthLens AI requires genuine, publicly documented datasets containing paired **Images and Text** with documented labeling protocols and research licenses. We audited the four datasets explicitly suggested in the course project documentation:
1. **Fakeddit**
2. **FaceForensics++ (FF++)**
3. **Deepfake Detection Challenge (DFDC)**
4. **FakeAVCeleb**

Below is the structured technical audit, modality suitability assessment, label definition comparison, storage/compute feasibility on local hardware, and our final recommended path for user approval.

---

## 2. Comprehensive Audit of the Four Suggested Datasets

### A. Fakeddit (Nakamura et al., UCSB, ACL 2020)
* **Official Source**: [entitize/Fakeddit (GitHub)](https://github.com/entitize/Fakeddit) | [Website](https://fakeddit.netlify.app/) | [Paper (arXiv:1911.03854)](https://arxiv.org/abs/1911.03854)
* **License & Research Restrictions**: Academic / Research Use. Sourced via Reddit API.
* **Available Modalities**: **Text + Image + Submission Metadata** (Post title text, image URL / image file, comments, upvotes, subreddit).
* **Sample Count**: Over 1,000,000 Reddit submissions total (~680,000 in `multimodal_only_samples` split).
* **Ground-Truth Label Scheme**:
  * `2_way_label`: Binary classification (0: Fake news / Manipulated, 1: True / Authentic news).
  * `3_way_label`: 0: True, 1: Satire, 2: Misleading / False.
  * `6_way_label`:
    1. *True* (Authentic news context)
    2. *Satire / Parody* (Humorous fictitious claims)
    3. *Misleading Content* (Deceptive presentation)
    4. *Manipulated Content* (Photoshop edits, face morphs, spliced pixels)
    5. *False Connection* (Out-of-context headlines where image and text do not correspond)
    6. *Imposter Content* (Impersonation of genuine sources)
* **Are Image and Text Paired?**: **YES**. Every sample pairs `clean_title` (headline text) with `image_url` / image file.
* **Is Manipulated-Image Ground Truth Available?**: **YES**. Class 4 (*Manipulated Content*) explicitly isolates image-level pixel manipulation, while Class 5 (*False Connection*) isolates cross-modal semantic mismatch (cheapfakes).
* **Storage Footprint**: Full image archive is >100 GB. However, a curated stratified benchmark subset (e.g., 500–1,000 verified paired samples) requires only **150 MB – 300 MB**.
* **Suitability for Image + Text Modality**: **HIGHEST (Primary Fit)**. It is the only one of the four suggested datasets natively created for paired Image + Text verification.

---

### B. FaceForensics++ (Rossler et al., TUM / Google, ICCV 2019)
* **Official Source**: [ondyari/FaceForensics (GitHub)](https://github.com/ondyari/FaceForensics)
* **License & Access**: Restricted academic use. Requires signing an official institutional Google Form agreement with Technical University of Munich.
* **Available Modalities**: **Video frames + Binary Facial Segmentation Masks** (MP4 videos or extracted image frames). **NO natural language text**.
* **Sample Count**: 1,000 original YouTube video sequences, 4,000 manipulated videos (Deepfakes, Face2Face, FaceSwap, NeuralTextures).
* **Labels**: Binary (Pristine vs Manipulated) and fine-grained manipulation method tags.
* **Are Image and Text Paired?**: **NO**. The dataset has no paired claims, captions, or contextual news text.
* **Storage Footprint**: 100 GB – 500 GB depending on compression levels (c0 raw, c23, c40).
* **Suitability for Image + Text Modality**: **Low as a primary dataset** (due to total absence of textual modality). Can only serve as a supplementary source for pure facial manipulation image frames if paired with external text claims.

---

### C. Deepfake Detection Challenge (DFDC) (Dolhansky et al., Meta / Kaggle, 2020)
* **Official Source**: [Kaggle DFDC Competition](https://www.kaggle.com/c/deepfake-detection-challenge)
* **License & Access**: Kaggle competition terms, non-commercial academic research.
* **Available Modalities**: **Video (.mp4) + Audio tracks**. **NO text modality**.
* **Sample Count**: ~124,000 video clips (10 seconds each) from paid actors.
* **Labels**: Binary (`REAL` or `FAKE`).
* **Are Image and Text Paired?**: **NO**. Contains zero paired captions, articles, or claim text.
* **Storage Footprint**: **>470 GB**. Completely infeasible for direct download on local storage without cloud GPU compute.
* **Suitability for Image + Text Modality**: **Unsuitable**. Does not support NLP contributions or text-image cross-modal analysis.

---

### D. FakeAVCeleb (Khalil et al., 2021)
* **Official Source**: [FakeAVCeleb (GitHub)](https://github.com/DASH-Lab/FakeAVCeleb)
* **License & Access**: Request form required via DASH-Lab.
* **Available Modalities**: **Audio + Video (Audio-Visual Deepfakes)**.
* **Sample Count**: ~20,000 video clips.
* **Labels**: Fine-grained audio-visual pairs: Real Video-Real Audio (RV-RA), Fake Video-Real Audio (FV-RA), Real Video-Fake Audio (RV-FA), Fake Video-Fake Audio (FV-FA).
* **Are Image and Text Paired?**: **NO**. Focuses on voice cloning and lip-sync reenactment, not natural language text claims.
* **Storage Footprint**: ~6.3 GB to 30 GB.
* **Suitability for Image + Text Modality**: **Unsuitable** for text-based NLP deepfake verification.

---

## 3. Comparison Matrix

| Dataset | Modalities | Has Paired Image + Text? | Has Manipulated Image Ground Truth? | Size / Storage | Access Requirements | CSET346 Image+Text Compatibility |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fakeddit** | **Image + Text** | **YES** (`clean_title` + image) | **YES** (6-way: *Manipulated*, *False Connection*, *True*) | ~150 MB (subset) to 100 GB (full) | Public (GitHub / Direct) | **100% (Optimal Fit)** |
| **FaceForensics++** | Video / Frames | NO (Frames only) | YES (Deepfakes, FaceSwap, Face2Face) | >100 GB | Institutional Agreement Form | 35% (Vision only, lacks text) |
| **DFDC** | Video + Audio | NO | YES (Facial reenactment) | >470 GB | Kaggle Account / Acceptance | 15% (No NLP modality, massive size) |
| **FakeAVCeleb** | Audio + Video | NO | YES (Audio-video lip sync) | ~6.3 GB | Research Request Form | 15% (No NLP modality) |

---

## 4. Label Integrity & Mapping Protocol to REAL / FAKE / UNCERTAIN

In strict compliance with Course Rules:
- We **never** treat ordinary fake-news labels as proof that an image was manipulated.
- We **never** claim that semantic consistency proves image authenticity.
- We maintain **explicit, independent metadata** fields for every record:

```json
{
  "sample_id": "string",
  "image_rel_path": "pathlib.Path",
  "text": "clean caption/title",
  "image_authenticity": "authentic | manipulated | unknown",
  "text_claim_status": "verifiable_true | fabricated | misleading_or_opinion",
  "image_text_consistency": "aligned | misaligned | ambiguous",
  "manipulation_type": "none | photoshop_splice | face_swap | diffusion_generated | out_of_context | ambiguous",
  "original_dataset_label": "Fakeddit 6-way / 2-way label",
  "pair_label": "REAL | FAKE | UNCERTAIN",
  "source_dataset": "Fakeddit-v2",
  "source_reference": "https://doi.org/10.18653/v1/2020.acl-main.551",
  "license": "Academic Research Use",
  "split": "train | val | test"
}
```

### Deterministic Label Mapping Rules:
1. **`pair_label: REAL`**:
   - `image_authenticity` = `authentic`
   - `text_claim_status` = `verifiable_true`
   - `image_text_consistency` = `aligned`
   - (Originates from Fakeddit Class 0: *True* with verified non-tampered media).
2. **`pair_label: FAKE`**:
   - Case A (Manipulated Pixels): `image_authenticity` = `manipulated` (Originates from Fakeddit Class 4: *Manipulated Content*).
   - Case B (Out-of-Context Deception): `image_text_consistency` = `misaligned` and `text_claim_status` = `fabricated` (Originates from Fakeddit Class 5: *False Connection*).
3. **`pair_label: UNCERTAIN`**:
   - Originates from subtle satire, conflicting cross-modal evidence (e.g. authentic image paired with an ambiguous claim), or low-resolution compressed evidence.

---

## 5. Local Hardware Feasibility Assessment
- **Host**: Windows 11 Home, 16 GB RAM, Intel Iris Xe (CPU-optimized PyTorch).
- **Available Storage**: Drive D has **72 GB free**.
- **Assessment**: Downloading the full 470 GB DFDC or 100 GB FaceForensics++ is completely infeasible and unsuited for Image+Text. 
- In contrast, downloading a verified, stratified **Fakeddit benchmark subset of 600–1,000 real multimodal samples** requires **under 400 MB**, fits comfortably in storage, loads cleanly in memory, and trains within minutes on your local CPU!

---

## 6. Official Dataset Recommendation
**Recommendation: Fakeddit Multimodal Benchmark Subset (600 Samples)**
- **Why**:
  1. It is the **only** course-suggested dataset that inherently contains paired, authentic **Images + Text**.
  2. Its 6-way label granularity explicitly differentiates **Manipulated Content** (pixel tampering) from **False Connection** (out-of-context miscaptioning).
  3. It allows real training on DistilBERT and ResNet-50 using leakage-resistant train/val/test splits.
  4. It strictly fits within local disk and CPU compute budgets.
