# DATASET CARD: Fakeddit Multimodal Benchmark Subset

## 1. Dataset Details
- **Dataset Name**: Fakeddit (Fine-grained Multimodal Misinformation & Deepfake Dataset)
- **Creators**: Kai Nakamura, Sharon Levy, William Yang Wang (UC Santa Barbara NLP Lab)
- **Paper**: *r/Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection*, ACL 2020.
- **Repository**: [https://github.com/entitize/Fakeddit](https://github.com/entitize/Fakeddit)
- **License**: Academic and Non-Commercial Research Use
- **Modality**: **Paired Image (.jpg/.png) + English Text Title/Caption**

---

## 2. Intended Use & Scope
- **Domain**: Multimodal fake content detection, visual manipulation forensics, and out-of-context miscaptioning analysis.
- **Target Task in TruthLens AI**: Joint image-text verification predicting `REAL`, `FAKE`, or `UNCERTAIN` with evidence-based explanations.

---

## 3. Data Structure & Fields
Each verified record contains:
- `id`: Unique Reddit post identifier
- `clean_title`: Normalized text caption / claim
- `image_path`: Local relative path to stored image
- `2_way_label`: Binary label (0: Fake/Manipulated, 1: True)
- `3_way_label`: 3-way label (0: True, 1: Satire, 2: False)
- `6_way_label`:
  - 0: True
  - 1: Satire / Parody
  - 2: Misleading Content
  - 3: Manipulated Content (Visual image tampering)
  - 4: False Connection (Cross-modal mismatch / out-of-context)
  - 5: Imposter Content
- `image_authenticity`: `authentic` | `manipulated` | `unknown`
- `text_claim_status`: `verifiable_true` | `fabricated` | `misleading_or_opinion`
- `image_text_consistency`: `aligned` | `misaligned` | `ambiguous`
- `pair_label`: `REAL` | `FAKE` | `UNCERTAIN`
- `split`: `train` (70%) | `val` (15%) | `test` (15%)

---

## 4. Data Quality & Limitations
- **Broken URLs Mitigation**: Live downloading from Reddit URLs can suffer from deleted posts (404s). Our builder downloads, verifies HTTP 200 responses, tests image header readability with PIL, and validates $(224 \times 224 \times 3)$ RGB tensors before admitting any sample into the final dataset.
- **Contextual Nuance**: Satire and parody often feature authentic images with absurd headlines; these are systematically mapped to `UNCERTAIN` or evaluated in Experiment 5 (Difficult-Case Analysis).
- **Domain Boundary**: Sourced from public discourse on Reddit; language reflects informal online discourse and news aggregations.
