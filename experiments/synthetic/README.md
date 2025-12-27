# Synthetic Experiment: Hidden Subgroup Discovery

This experiment reproduces the synthetic artifact discovery scenario. We inject a "vertical line" artifact into a specific class of the Imagenette dataset and check if subgroup discovery can identify it as a high-error or distinct subgroup.

## Reproduction Steps (Colab / Local)

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Data**:
   Download Imagenette (160px version).
   ```bash
   python scripts/1_setup_data.py
   ```

3. **Generate Synthetic Dataset**:
   Inject artifacts into the target class.
   ```bash
   python scripts/2_generate_synthetic.py
   ```

4. **Train Classification Model**:
   Train a ResNet model on the biased dataset.
   ```bash
   python scripts/3_train_model.py --epochs 10 --project "synthetic_experiment"
   ```

5. **Extract Features**:
   Get CLIP embeddings for the images.
   ```bash
   python scripts/4_extract_features.py
   ```

6. **Run Analysis**:
   Discover subgroups and verify artifact identification.
   ```bash
   python scripts/5_run_analysis.py
   ```

## M1 MacBook Support

If you are running on an M1 MacBook, use the provided configuration for optimal performance:
```bash
python scripts/3_train_model.py --config configs/m1_config.yaml
python scripts/4_extract_features.py --config configs/m1_config.yaml
python scripts/5_run_analysis.py --config configs/m1_config.yaml
```

## Folder Structure

- `src/`: Core modular logic (data, model, analysis).
- `scripts/`: Orchestration scripts for the experiment pipeline.
- `notebooks/`: Interactive reproduction (Colab-ready).
- `configs/`: Experiment hyperparameters.
- `requirements.txt`: Python package dependencies.
