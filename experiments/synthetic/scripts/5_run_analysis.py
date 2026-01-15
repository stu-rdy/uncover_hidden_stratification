import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import meerkat as mk
import wandb
from domino import DominoSlicer

# Add project root and experiment src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../"))
sys.path.append(PROJECT_ROOT)

from experiments.synthetic.src.analysis import analyze_slices
from experiments.synthetic.scripts.analysis_utils import (
    plot_slice_performance,
    plot_error_concentration,
    extract_slice_examples,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--predictions_csv",
        type=str,
        default=os.path.join(PROJECT_ROOT, "saves/synthetic/test_predictions.csv"),
    )
    parser.add_argument("--n_slices", type=int, default=10)
    parser.add_argument("--weight", type=float, default=10.0)
    parser.add_argument("--project", default="synthetic_imagenette")
    parser.add_argument("--extract_examples", action="store_true", default=True)
    parser.add_argument("--no_extract", action="store_false", dest="extract_examples")
    parser.add_argument("--n_examples", type=int, default=5)
    args = parser.parse_args()

    # Load YAML config if provided
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    # Merge args with config
    n_slices = (
        args.n_slices
        if not config.get("analysis")
        else config["analysis"].get("n_slices", args.n_slices)
    )
    weight = (
        args.weight
        if not config.get("analysis")
        else config["analysis"].get("weight", args.weight)
    )

    # WandB Init
    wb_conf = config.get("wandb", {})
    project = wb_conf.get("project", args.project)

    wandb.init(
        project=project,
        entity=wb_conf.get("entity"),
        tags=wb_conf.get("tags"),
        notes=wb_conf.get("notes"),
        job_type="analysis",
        config=config or args,
    )

    data_root = os.path.join(PROJECT_ROOT, "data/synthetic_imagenette")
    val_embed_path = os.path.join(data_root, "val_embeddings.mk")
    test_embed_path = os.path.join(data_root, "test_embeddings.mk")

    if not os.path.exists(val_embed_path):
        print(
            f"Val embeddings not found at {val_embed_path}. Run 4_extract_features.py first."
        )
        wandb.finish()
        return
    if not os.path.exists(test_embed_path):
        print(
            f"Test embeddings not found at {test_embed_path}. Run 4_extract_features.py first."
        )
        wandb.finish()
        return

    print("Loading data...")
    df_val = mk.read(val_embed_path)
    df_test = mk.read(test_embed_path)

    # Check for NaNs in embeddings
    if np.isnan(df_val["clip(img)"].data).any():
        print("⚠️ Warning: NaNs found in validation embeddings! Filling with 0.")
        df_val["clip(img)"].data = np.nan_to_num(df_val["clip(img)"].data)

    if np.isnan(df_test["clip(img)"].data).any():
        print("⚠️ Warning: NaNs found in test embeddings! Filling with 0.")
        df_test["clip(img)"].data = np.nan_to_num(df_test["clip(img)"].data)

    # Sanity check: ensure embeddings were extracted correctly
    assert "clip(img)" in df_val.columns, (
        f"Expected 'clip(img)' column in embeddings. Found: {list(df_val.columns)}"
    )
    assert "clip(img)" in df_test.columns, (
        f"Expected 'clip(img)' column in embeddings. Found: {list(df_test.columns)}"
    )

    # Load predictions for val and test
    val_preds_path = os.path.join(PROJECT_ROOT, "saves/synthetic/val_predictions.csv")
    test_preds_path = args.predictions_csv  # test_predictions.csv

    val_preds_df = pd.read_csv(val_preds_path)
    test_preds_df = pd.read_csv(test_preds_path)

    # Merge predictions by image name (not index position - avoids silent misalignment)
    pred_cols = [c for c in val_preds_df.columns if c.startswith("pred_")]
    sorted_pred_cols = sorted(pred_cols, key=lambda x: int(x.split("_")[1]))
    n_classes = len(sorted_pred_cols)

    # Create lookup dicts keyed by image name for safe merging
    val_pred_lookup = {
        row["name"]: row[sorted_pred_cols].values.astype(np.float64)
        for _, row in val_preds_df.iterrows()
    }
    test_pred_lookup = {
        row["name"]: row[sorted_pred_cols].values.astype(np.float64)
        for _, row in test_preds_df.iterrows()
    }

    # Match predictions to embeddings by name
    val_probs = np.array(
        [val_pred_lookup[name] for name in df_val["image_path"].data], dtype=np.float64
    )
    test_probs = np.array(
        [test_pred_lookup[name] for name in df_test["image_path"].data],
        dtype=np.float64,
    )

    # Convert targets to one-hot for stability in Domino if needed
    val_targets = np.eye(n_classes)[df_val["target"].data.astype(int)]
    test_targets = np.eye(n_classes)[df_test["target"].data.astype(int)]

    # Normalize prediction probabilities to ensure they sum to 1 and have no zeros
    val_probs = np.clip(val_probs, 1e-6, 1.0)
    val_probs /= val_probs.sum(axis=1, keepdims=True)
    test_probs = np.clip(test_probs, 1e-6, 1.0)
    test_probs /= test_probs.sum(axis=1, keepdims=True)

    df_val["pred_probs"] = val_probs
    df_test["pred_probs"] = test_probs
    df_val["target_onehot"] = val_targets
    df_test["target_onehot"] = test_targets

    # Check for NaNs in prediction probabilities
    if np.isnan(val_probs).any() or np.isnan(test_probs).any():
        print("⚠️ Warning: NaNs found in prediction probabilities!")

    # Set environment variable to avoid KMeans memory leak on Windows
    os.environ["OMP_NUM_THREADS"] = "1"

    # Run Domino - FIT ON VAL, predict on both (matches original notebook methodology)
    print(f"Running Domino with {n_slices} slices and weight {weight}...")
    domino = DominoSlicer(
        y_log_likelihood_weight=0,
        y_hat_log_likelihood_weight=weight,
        n_mixture_components=25,
        n_slices=n_slices,
        confusion_noise=1e-3,
        max_iter=100,
        random_state=42,
    )

    # Fit on validation data (no peeking at test)
    domino.fit(
        data=df_val,
        embeddings="clip(img)",
        targets="target_onehot",
        pred_probs="pred_probs",
    )

    # Predict on both val and test
    df_val["domino_slices"] = domino.predict(
        data=df_val,
        embeddings="clip(img)",
        targets="target_onehot",
        pred_probs="pred_probs",
    )
    df_test["domino_slices"] = domino.predict(
        data=df_test,
        embeddings="clip(img)",
        targets="target_onehot",
        pred_probs="pred_probs",
    )

    # Extract hard slice assignments (domino_slices is a soft assignment matrix [n_samples, n_slices])
    slice_preds_val = np.argmax(df_val["domino_slices"].data, axis=1)
    slice_preds_test = np.argmax(df_test["domino_slices"].data, axis=1)

    # Get predicted class labels from probabilities
    val_pred_labels = np.argmax(df_val["pred_probs"].data, axis=1)
    test_pred_labels = np.argmax(df_test["pred_probs"].data, axis=1)

    # Analysis on both val and test (now includes per-slice accuracy)
    print("\n=== Validation Set Analysis ===")
    val_res_df = analyze_slices(
        slice_preds_val,
        df_val["target"].data,
        val_pred_labels,
        metadata={
            "hidden": df_val["has_hidden_artifact"].data,
            "known": df_val["has_known_artifact"].data,
        },
    )
    val_res_df["split"] = "val"
    print(val_res_df)

    print("\n=== Test Set Analysis ===")
    test_res_df = analyze_slices(
        slice_preds_test,
        df_test["target"].data,
        test_pred_labels,
        metadata={
            "hidden": df_test["has_hidden_artifact"].data,
            "known": df_test["has_known_artifact"].data,
        },
    )
    test_res_df["split"] = "test"
    print(test_res_df)

    # Combine results
    res_df = pd.concat([val_res_df, test_res_df], ignore_index=True)

    print("\nSlice Analysis Summary:")
    print(res_df)

    out_path = os.path.join(PROJECT_ROOT, "results/synthetic_analysis.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    res_df.to_csv(out_path, index=False)
    print(f"\nSaved detailed analysis to {out_path}")

    # Log results to wandb
    table = wandb.Table(dataframe=res_df)
    wandb.log({"analysis_results": table})

    # Log per-slice accuracy bar charts for test set (key visualization)
    if "accuracy" in test_res_df.columns:
        # Per-slice accuracy bar chart
        acc_data = [
            [f"Slice {row['slice']}", row["accuracy"]]
            for _, row in test_res_df.iterrows()
        ]
        acc_table = wandb.Table(data=acc_data, columns=["slice", "accuracy"])
        wandb.log(
            {
                "test/slice_accuracy": wandb.plot.bar(
                    acc_table, "slice", "accuracy", title="Test Set: Per-Slice Accuracy"
                )
            }
        )

        # Per-slice hidden artifact rate
        hidden_data = [
            [f"Slice {row['slice']}", row["hidden_rate"]]
            for _, row in test_res_df.iterrows()
        ]
        hidden_table = wandb.Table(data=hidden_data, columns=["slice", "hidden_rate"])
        wandb.log(
            {
                "test/slice_hidden_rate": wandb.plot.bar(
                    hidden_table,
                    "slice",
                    "hidden_rate",
                    title="Test Set: Per-Slice Hidden Artifact Rate",
                )
            }
        )

        # Per-slice known artifact rate
        known_data = [
            [f"Slice {row['slice']}", row["known_rate"]]
            for _, row in test_res_df.iterrows()
        ]
        known_table = wandb.Table(data=known_data, columns=["slice", "known_rate"])
        wandb.log(
            {
                "test/slice_known_rate": wandb.plot.bar(
                    known_table,
                    "slice",
                    "known_rate",
                    title="Test Set: Per-Slice Known Artifact Rate",
                )
            }
        )

        # Log key summary metrics
        wandb.log(
            {
                "test/worst_slice_accuracy": test_res_df["accuracy"].min(),
                "test/accuracy_gap": test_res_df["accuracy"].max()
                - test_res_df["accuracy"].min(),
            }
        )

    # --- Integrated Visualization and Extraction ---
    print("\n=== Generating Integrated Visualizations and Examples ===")
    plot_dir = os.path.join(PROJECT_ROOT, "results/plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Paths for plots
    perf_plot_path = os.path.join(plot_dir, "slice_analysis.png")
    conc_plot_path = os.path.join(plot_dir, "error_concentration.png")

    # Generate Plots
    plot_slice_performance(test_res_df, perf_plot_path)
    plot_error_concentration(test_res_df, conc_plot_path)

    # Extract Examples
    if args.extract_examples:
        example_dir = os.path.join(PROJECT_ROOT, "results/slice_examples")

        # Prepare df_test for extraction (add slice assignments and predictions)
        # Select only necessary columns to avoid conversion warnings for Meerkat tensor columns
        cols_to_keep = ["image_path", "target"]
        pd_test = df_test[cols_to_keep].to_pandas()
        pd_test["domino_slice"] = slice_preds_test
        pd_test["prediction"] = test_pred_labels

        extract_slice_examples(
            pd_test, test_res_df, example_dir, n_examples=args.n_examples
        )

    wandb.finish()


if __name__ == "__main__":
    main()
