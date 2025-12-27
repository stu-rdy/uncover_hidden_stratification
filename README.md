# Subgroup Performance Analysis in Hidden Stratifications (under review for MICCAI 2025)

## Abstract

Machine learning (ML) models may suffer from significant performance disparities between patient groups.
Identifying such disparities by monitoring performance at a granular level is crucial for safely deploying ML to each patient.
Traditional subgroup analysis based on metadata can expose performance disparities only if the available metadata (e.g. patient sex) sufficiently reflects the main reasons for performance variability, which is not common.
Subgroup discovery techniques that identify cohesive subgroups based on learned feature representations appear as a potential solution: They could expose hidden stratifications and provide more granular subgroup performance reports.
However, subgroup discovery is challenging to evaluate even as a standalone task, as ground truth stratification labels do not exist in real data. Subgroup discovery has thus neither been applied nor evaluated for the application of subgroup performance monitoring. Here, we apply subgroup discovery for performance monitoring in chest x-ray and skin lesion classification. We propose novel evaluation strategies and show that a simplified subgroup discovery method without access to classification labels or metadata can expose larger performance disparities than traditional metadata-based subgroup analysis. We provide the first compelling evidence that subgroup discovery can serve as an important tool for comprehensive performance validation and monitoring of trustworthy AI in medicine.

## Datasets
Metadata-rich open datasets:
CheXpertPlus: A metadata augmented version of CheXpert, which is a large dataset of chest x-ray images.
SLICE-3D: 400,000 skin lesion image crops extracted from 3D TBP for skin cancer detection.

## How to Run

### Step 1: Train the Target Classification Models
- Run classification training scripts in `experiments/`:
  - `python experiments/chest_xray/train_ERM-cxr.py`
  - `python experiments/skin_lesion/train_ERM-skin.py`
- Set the desired learning rate (`--lr`) and weight decay (`--wd`) as needed.
- The project is integrated with [Weights & Biases](https://wandb.ai/) for experiment tracking.

### Step 2: Extract Features with CLIP (or Other External Models)
- Run extraction notebooks in `experiments/`:
  - `experiments/chest_xray/domino_cxr.ipynb`
  - `experiments/skin_lesion/domino_isic.ipynb`
- The main function `embed` is imported from the `domino` module.
- The encoder defines the model used for feature extraction.
- The output is a Meerkat DataFrame, saved for the next step.

### Step 3: Fit the Subgroup Discovery Method
- Run `notebooks/quantitative_analysis.ipynb`.
- The functions `run_domino_*` learn subgroup divisions based on:
  - The desired number of subgroups (`n_slices`).
  - The hyperparameter \(\gamma\) (controlled by `y_hat_log_likelihood_weight`).
- The function `eval_subgroup_statistics(df)` evaluates subgroup statistics using the output from `run_domino_*`.
- The notebook concludes by generating the necessary CSV files for plotting.

### Step 4: Plot the Results
- Reproduce all plots using the notebooks in `notebooks/plots/`.
