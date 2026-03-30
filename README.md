#  Green vs Non-Green Architecture Classification (EcoArch-SAFNet)

This repository provides overview of our deep learning-based framework for classifying sustainable (Green) vs. non-sustainable (Non-Green) architectural designs.

##  Overview

The study applies tabular deep learning and explainable AI (XAI) techniques to analyze architectural features across four sustainability categories:

- **Energy**
- **Eco-Tech**
- **Design**
- **Context**

The proposed model incorporates attention mechanisms and fusion layers to enhance interpretability and performance.

##  Contents

- `green_architecture_dataset.csv`: Input dataset.
- `Green_Architecture_Classification_Full.ipynb`
- Real-time visualizations:
  - Confusion matrices
  - Attention heatmaps
  - SHAP & LIME plots
  - Information gain & reliability curves

##  Requirements

- Python 3.8+
- PyTorch
- Scikit-learn
- LIME
- SHAP
- Matplotlib, Seaborn, Pandas

Install all dependencies:
```bash
pip install -r requirements.txt
