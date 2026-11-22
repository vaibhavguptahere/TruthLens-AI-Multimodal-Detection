# TruthLens AI: Presentation Slide Deck (Milestone Final Defense)

**Course**: CSET346 – Natural Language Processing  
**Project**: Multimodal NLP for Deepfake Detection & Explainable Content Verification  
**Platform**: Local Windows PC Implementation  

---

## Slide 1: Title & Team
- **Title**: TruthLens AI: Multimodal Deepfake Detection & Explainable Content Verification
- **Course**: CSET346 Natural Language Processing
- **Focus**: Paired Image + Text Verification (REAL, FAKE, UNCERTAIN)
- **Key Tenet**: Decoupling Visual Manipulation from Textual Deception

---

## Slide 2: Problem Statement & Motivation
- **The Challenge**: Misinformation is rarely unimodal.
  - Spliced or AI-generated images often carry truthful text.
  - Authentic photos are frequently weaponized with fabricated claims (cheapfakes / out-of-context).
- **Core Research Question**:
  *"Can multimodal NLP and vision models jointly identify deepfakes and context mismatches better than single-modality baselines while providing evidence-based explanations?"*

---

## Slide 3: Decoupled Multimodal Formulation
- **Why Ordinary Fake-News Labels Fail**:
  - An authentic photo with a misleading caption does not make the photo a deepfake.
- **Our Five-Part Metadata Schema**:
  1. `image_authenticity`: Authentic vs Manipulated
  2. `text_claim_status`: Verifiable True vs Fabricated
  3. `image_text_consistency`: Aligned vs Misaligned
  4. `manipulation_type`: Diffusion, FaceSwap, Splice, Out-of-Context
  5. `pair_label`: **REAL**, **FAKE**, or **UNCERTAIN**

---

## Slide 4: System Architecture
- **Dual Encoders**:
  - Text: **DistilBERT-base-uncased** ($768$-dim pooled embeddings)
  - Image: **ResNet-50** ($2048$-dim spatial feature bottleneck)
- **Two Fusion Strategies Tested**:
  1. *Feature-Level Fusion*: Projections into common $256$-dim bottleneck $\to$ Concatenation $\to$ MLP Classifier.
  2. *Decision-Level Fusion*: Independent calibrated softmax probabilities $\to$ Polar conflict estimation $\to$ Dynamic routing to UNCERTAIN.

---

## Slide 5: The Three-Class Protocol & Uncertainty
- **Why UNCERTAIN is Essential**:
  - Softmax models often produce $99\%$ overconfidence on ambiguous inputs.
- **Our Calibrated Protocol**:
  - Divergent modalities (Image says FAKE, Text says REAL) $\to$ High Conflict Score ($>0.40$) $\to$ Automatically routed to **UNCERTAIN**.
  - Flat probability distributions (Low peak confidence $<0.60$) $\to$ Routed to **UNCERTAIN**.

---

## Slide 6: Dual Explainability Pipeline
- **Visual Evidence (Grad-CAM)**:
  - Hooks into ResNet-50's `layer4` conv stage.
  - Generates spatial heatmaps showing exactly which pixel regions triggered the verdict.
- **Textual Evidence (Token Saliency)**:
  - Backpropagates gradient norms to word embeddings.
  - Highlights specific deceptive or influential words in captions.
- **Cross-Modal Consistency**:
  - Cosine alignment between projected embeddings detects out-of-context miscaptioning.

---

## Slide 7: Experimental Benchmarks (Key Findings)
- **Experiment 1 (Single vs Multimodal)**:
  - Text-Only Baseline: $54.4\%$ Acc, $0.379$ F1
  - Image-Only Baseline: $44.1\%$ Acc, $0.204$ F1
  - Multimodal Model: Balances cues across both streams.
- **Experiment 2 (Fusion Comparison)**:
  - Feature-level fusion preserves cross-modal correlation.
  - Decision-level fusion provides superior conflict filtering.
- **Experiment 6 (Explainability Faithfulness)**:
  - Grad-CAM concentration ratios: $27.6 \times$ to $48.2 \times$ peak localization.
  - $100\%$ of evaluated samples verified as `FAITHFUL`.

---

## Slide 8: Difficult-Case Analysis (15 Corner Cases)
- Tested 15 specific hard samples:
  - Subtle Photoshop seams.
  - Authentic photos with deceptive breaking-news claims.
  - High-compression blurred thumbnails.
- **Result**: In cases of conflicting evidence, the system avoided false positive certainty and correctly triggered the **UNCERTAIN** protocol with explicit natural language justification.

---

## Slide 9: Full-Stack Demonstration
- **FastAPI REST API**:
  - `POST /api/verify` accepting multipart image and caption text.
  - Returns structured JSON containing verdict, confidence, base64 Grad-CAM heatmap, and token importance scores.
- **React 18 User Interface**:
  - Interactive web dashboard with image drag-and-drop, verdict banners, and interactive Grad-CAM heatmap overlay toggle.

---

## Slide 10: Conclusion & Future Outlook
- **Key Accomplishments**:
  - Fully local, reproducible pipeline on Windows 11 PC.
  - Zero fabricated numbers; 100% adherence to CSET346 guidelines.
  - All 20 traceability requirements tested and validated.
- **Future Extensions**:
  - Cross-attention transformer fusion.
  - Video temporal frame analysis.
