# Reproduction Report: Hidden Subgroup Performance Discovery

This document details the reproduction of the synthetic experiment methodology described in **Bissoto et al. (2024), "Subgroup Performance Analysis in Hidden Stratifications"**.

## 1. Objective
The goal was to demonstrate how "shortcut learning" leads to massive performance disparities in hidden patient subgroups, and how these failures can be automatically discovered using the **Domino** algorithm even when the "average" accuracy of a model remains high.

## 2. Technical Modifications & Implementations

### A. Dynamic Artifact Alignment (`src/data.py`)
To ensure visual consistency for Keynote presentations and professional-grade synthetic data:
- Implemented `cv2.getTextSize` logic to dynamically scale font size.
- Centered "ID - TAG" text within the white hospital tag artifact.
- Fixed clipping issues where the label exceeded the artifact boundaries.

### B. "Uglier" Data Pipeline (X-Ray Simulation)
To simulate the difficulty of medical imaging vs. natural photos, we added a degradation pipeline to the base Imagenette images:
- **Grayscale Conversion**: Mimics X-ray/monochrome medical modalities.
- **Gaussian Blur ($\sigma=2.0$)**: Simulates motion blur or low-resolution sensors.
- **Gaussian Noise ($\text{std}=0.03$)**: Simulates electronic sensor grain.
- **Result**: The "real" features (fish) became harder to learn, while the "shortcut" feature (ID-TAG) remained sharp and salient.

### C. Training & Early Stopping (`scripts/3_train_model.py`)
- **Metric Monitoring**: Updated the trainer to monitor **Worst-Group Accuracy (WGA)** rather than average Accuracy.
- **Early Stopping**: Implemented logic to save the model where WGA is highest, preventing the model from over-relying on shortcuts as training progresses.
- **No Pretraining**: Disabled ImageNet weights to force the model to learn features from the noisy/blurry distribution from scratch.

## 3. Alignment with Bissoto et al. (MICCAI)

| Parameter | Paper Description | Our Reproduction | Rationale |
| :--- | :--- | :--- | :--- |
| **Bias Level ($p$)** | 0.6, 0.7, 0.8 | **0.8** | Maximum bias tested in the paper. |
| **Slicing Resolution**| 15 Slices | **15** | Standardized for 10-class discovery. |
| **Slicing Weight ($\gamma$)**| 10 | **10** | Balances artifact discovery vs. group cohesion. |
| **Artifact Location** | Bottom-left tag | **Bottom-left tag** | Identical visual shortcut usage. |
| **Evaluation Tool** | Domino | **Domino** | Identical analysis algorithm. |

## 4. Key Results & Insights

### The "Shortcut Gap"
In the final run, we observed:
- **Training Accuracy**: ~98% (The model "solves" the shortcut).
- **Test Accuracy (Avg)**: ~71% (The model looks acceptable).
- **Worst-Group Acc (WGA)**: **~40%** (The model is failing on a specific subgroup).

### Discovery Success
**Domino** successfully isolated the failure mode in **Slice 0**.
- **Slice 0 Characteristics**: 95% Artifact Rate, 40% Accuracy.
- **Interpretation**: This subgroup captures the images where the model was successfully "poisoned" by the shortcut. In a clinical setting, this would represent a group of patients for whom the AI is systematically wrong because it is looking at a label rather than the pathology.

## 5. Conclusion
This reproduction confirms the paper's core finding: **Deep learning models are "lazy learners."** In medical domains like X-ray analysis, they will prioritize high-contrast shortcuts (like hospital ID tags) over subtle biological features (like cardiomegaly) if the latter are even slightly harder to detect. This failure is invisible in "Average Accuracy" but glaringly obvious when using subgroup discovery.
