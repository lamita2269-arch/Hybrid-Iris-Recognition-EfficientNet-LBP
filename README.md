# Hybrid-Iris-Recognition-EfficientNet-LBP
Proposed hybrid iris recognition model combining EfficientNet-B1 and LBP features with SVM classification


## Overview
This repository contains the official implementation of a hybrid iris recognition framework. The system combines deep feature extraction via **EfficientNet-B1** with traditional texture analysis using **Local Binary Patterns (LBP)**, classified by a fine-tuned **SVM**.

## Key Achievements
- **Accuracy:** 97.53%
- **AUC Score:** 0.9982
- **Architecture:** Hybrid Fusion (Deep Learning + Handcrafted Features)

## Repository Structure
- `iris_93_plus.keras`: Pre-trained weights for the deep learning branch.
- `Hybrid_Iris_Recognition.ipynb`: Complete pipeline from preprocessing to evaluation.
- `results/`: Visualization of Confusion Matrix, ROC curves, and performance metrics.

## Methodology
The proposed model follows a multi-stage pipeline:
1. Preprocessing & Normalization.
2. Deep Feature Extraction (EfficientNet-B1).
3. Texture Feature Extraction (LBP).
4. Feature Fusion & Standardization.
5. Classification via RBF-Kernel SVM.
   - `iris_93_plus.keras`: [Download Pre-trained Model Weights]https://drive.google.com/file/d/1v63dK3gKC6hL21j4-S7EEUGlmfeujHF4/view?usp=drive_linkا)
