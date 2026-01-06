# Synthetic Experiment: Hidden Stratification with Artificial Artifacts

This experiment reproduces the synthetic artifact methodology from **Bissoto et al. "Subgroup Performance in Hidden Stratifications"** using the Imagenette dataset.

---

## 1. Experimental Setup

### Dataset Construction
- **Base Dataset**: Imagenette (10 classes, 224×224 images)
- **Synthetic Artifact**: Gaussian noise patch applied to one corner of images
- **Biased Class**: Class 0 receives artifacts with 95% probability
- **Other Classes**: Receive artifacts with only 5% probability

### Train/Val/Test Split Design
| Split | Artifact Rate | Purpose |
|-------|---------------|---------|
| Train | ~13.7% | Biased distribution (class 0 dominates artifact presence) |
| Val   | ~51.0% | Decorrelated (artifacts uniformly distributed across classes) |
| Test  | ~50.0% | Decorrelated (same as val, for final evaluation) |

> [!IMPORTANT]
> The key design: training data has **spurious correlation** between artifacts and class 0, while evaluation data has **no correlation**. This forces the model to rely on shortcuts during training that fail at test time for specific subgroups.

### Model & Training
- **Architecture**: ResNet-18 (pretrained on ImageNet)
- **Optimizer**: SGD with momentum 0.9
- **Early Stopping**: Patience 5 on `val_acc`
- **Result**: 97.28% test accuracy (high overall, but hides subgroup issues)

---

## 2. What DOMINO Finds

DOMINO (Slice Discovery via Mixture Models) identifies **coherent subgroups** in the embedding space that share similar model behavior.

### Slice Discovery Results (Test Set)
| Slice | Size | Dominant Class | Class Purity | Accuracy |
|-------|------|----------------|--------------|----------|
| 0 | 195 | 3 | 97.4% | **92.3%** ← Worst |
| 1 | 182 | 0 | 100% | 98.9% |
| 2 | 192 | 8 | 100% | 98.9% |
| 3 | 221 | 5 | 98.2% | 97.3% |
| 4 | 205 | 7 | 98.0% | 96.6% |
| 5 | 200 | 9 | 98.5% | 98.0% |
| 6 | 213 | 1 | 100% | **99.5%** ← Best |
| 7 | 201 | 4 | 98.5% | 97.5% |
| 8 | 203 | 6 | 99.0% | 98.0% |
| 9 | 179 | 2 | 98.9% | 97.8% |

### Key Observations

1. **Slices align with classes** — Each discovered slice is dominated by a single class (97-100% purity), confirming DOMINO captures semantically meaningful groups.

2. **Performance varies by slice** — Accuracy ranges from 92.3% to 99.5%, a **7.2% gap** that is invisible in the aggregate 97.3% test accuracy.

3. **Slice 0 (Class 3) is the worst performer** — Despite having ~46% artifact rate (balanced), this slice has the lowest accuracy. This warrants investigation into what makes class 3 harder.

4. **Artifact rate is decorrelated at test time** — All slices have artifact rates between 42-55%, confirming the evaluation distribution is balanced as designed.

---

## 3. What the Metrics Reveal

### Summary Metrics
| Metric | Value | Interpretation |
|--------|-------|----------------|
| `test/worst_slice_accuracy` | 92.3% | Minimum accuracy across discovered slices |
| `test/accuracy_gap` | 7.2% | Spread between best and worst slice |
| Overall Test Accuracy | 97.3% | Aggregated performance (hides subgroup issues) |

### Training Distribution Metrics
| Metric | Expected | Actual |
|--------|----------|--------|
| `train/data/artifact_rate` | ~50% (naive) or ~14% | 13.7% |
| `train/data/artifact_rate/class_0` | ~95% | High (biased class) |
| `train/data/artifact_rate/class_X` | ~5% | Low (other classes) |
| `train/data/artifact_class_correlation` | High positive | Shows shortcut signal strength |

> [!NOTE]
> The training artifact rate is low (~14%) because only class 0 (10% of data) has 95% artifacts. This creates a spurious correlation: **artifact presence strongly predicts class 0**.

---

## 4. Interpretation

### Why This Matters
The model achieves **97.3% overall accuracy**, which appears excellent. However:

1. **Hidden Stratification Exists**: Slice 0 has 5% lower accuracy than the average.
2. **Aggregate Metrics Hide Harm**: A practitioner looking only at overall accuracy would miss this.
3. **DOMINO Surfaces the Problem**: Without slice discovery, the underperforming subgroup would remain invisible.

### Limitations of This Experiment
- **Synthetic Artifacts**: Real-world spurious correlations are rarely this clean.
- **Known Ground Truth**: We injected the artifact, so we know what to look for. In practice, the hidden stratification is truly hidden.
- **Balance at Test Time**: Real datasets may have imbalanced evaluation sets too.

---

## 5. Reproducing This Experiment

```bash
# 1. Setup data
python scripts/1_setup_data.py

# 2. Generate synthetic artifacts
python scripts/2_generate_synthetic.py --config configs/colab_config.yaml

# 3. Train model
python scripts/3_train_model.py --config configs/colab_config.yaml

# 4. Extract CLIP embeddings
python scripts/4_extract_features.py --config configs/colab_config.yaml

# 5. Run DOMINO slice discovery
python scripts/5_run_analysis.py --config configs/colab_config.yaml
```

Results are logged to [Weights & Biases](https://wandb.ai) and saved to `results/synthetic_analysis.csv`.

---

## References

- Bissoto, A., et al. "Subgroup Performance in Hidden Stratifications." *NeurIPS 2022.*
- Eyuboglu, S., et al. "Domino: Discovering Systematic Errors with Cross-Modal Embeddings." *ICLR 2022.*
