# Synthetic Experiment: Hidden Subgroup Discovery

This experiment reproduces the synthetic artifact discovery scenario based on Bissoto et al. We inject a "confounding" artifact into a specific class of the Imagenette dataset (320px) to simulate hidden stratification, and check if subgroup discovery can identify it as a high-error or distinct subgroup.

## Experiment Overview

- **Dataset**: Imagenette (320px).
- **Hidden Artifact**: "Hospital Tag" (injected into Class 0 with 80% probability vs 5% for others). This is the *hidden stratification* feature we want the model to rely on, which will cause failures where it is absent.
- **Known Artifact**: "Vertical Line" (injected with 25% probability across all classes). This is a known confounder tracked in metadata.
- **Modifications**: Images are modified with random Gaussian blur (sigma=2.0), noise (std=0.03), and grayscale conversion (to simulate X-rays).

## Reproduction Steps (Local)

> **Recommended**: For an interactive evaluation, use the [`reproduce_synthetic.ipynb`](experiments/synthetic/notebooks/reproduce_synthetic.ipynb) notebook.

For command-line reproduction:

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Data**:
   Download Imagenette (320px version).
   ```bash
   python experiments/synthetic/scripts/1_setup_data.py 
   ```

3. **Generate Synthetic Dataset**:
   Inject artifacts into the target class.
   ```bash
   python experiments/synthetic/scripts/2_generate_synthetic.py --no-wandb
   ```

4. **Train Classification Model**:
   Train a ResNet model on the biased dataset.
   ```bash
   python experiments/synthetic/scripts/3_train_model.py --epochs 15 --no-wandb
   ```

5. **Extract Features**:
   Get CLIP embeddings for the images.
   ```bash
   python experiments/synthetic/scripts/4_extract_features.py
   ```

6. **Run Analysis**:
   Discover subgroups and verify artifact identification.
   ```bash
   python experiments/synthetic/scripts/5_run_analysis.py --no-wandb
   ```

## Results

After running the analysis, results are saved in `results/`:

- `results/synthetic_analysis.csv`: Detailed metrics for discovered slices.
- `results/plots/`: Visualizations including slice performance and error concentration.
- `results/slice_examples/`: Saved images from discovered slices (if enabled).

## Folder Structure

- `experiments/synthetic/src/`: Core modular logic (data, model, analysis).
- `experiments/synthetic/scripts/`: Orchestration scripts for the experiment pipeline.
- `experiments/synthetic/notebooks/`: Interactive reproduction (Colab-ready).
- `experiments/synthetic/configs/`: Experiment hyperparameters.
- `requirements.txt`: Python package dependencies.
