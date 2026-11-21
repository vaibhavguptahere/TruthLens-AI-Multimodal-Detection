# Model Architecture Documentation

## 1. Overview
TruthLens AI implements a modular, comparative machine learning framework designed for local execution.

## 2. Components
### A. Text-Only Baseline
- **Backbone**: DistilBERT (`distilbert-base-uncased`) / BERT-base.
- **Head**: Linear projection + Softmax (REAL, FAKE, UNCERTAIN).

### B. Image-Only Baseline
- **Backbone**: ResNet-50 / ViT-B/16.
- **Head**: Global Average Pooling + Fully Connected layers + Softmax.

### C. Feature-Level Fusion (Multimodal Model 1)
- Extracted intermediate embeddings from Text and Image encoders.
- Projection layers to align dimensions.
- Concatenation or Cross-Attention fusion block.
- Joint classification head.

### D. Decision-Level Fusion (Multimodal Model 2)
- Independent inference from Image and Text baselines.
- Softmax probability aggregation (learned meta-classifier or calibrated weighted sum).
- Dynamic uncertainty thresholding for conflicting probabilities.
