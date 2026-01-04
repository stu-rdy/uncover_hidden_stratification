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

    # Load predictions for val and test
    val_preds_path = os.path.join(PROJECT_ROOT, "saves/synthetic/val_predictions.csv")
    test_preds_path = args.predictions_csv  # test_predictions.csv

    val_preds_df = pd.read_csv(val_preds_path)
    test_preds_df = pd.read_csv(test_preds_path)

    # Merge predictions
    pred_cols = [c for c in val_preds_df.columns if c.startswith("pred_")]
    sorted_pred_cols = sorted(pred_cols, key=lambda x: int(x.split("_")[1]))

    df_val["pred_probs"] = val_preds_df[sorted_pred_cols].values
    df_test["pred_probs"] = test_preds_df[sorted_pred_cols].values

    # Run Domino - FIT ON VAL, predict on both (matches original notebook methodology)
    print(f"Running Domino with {n_slices} slices and weight {weight}...")
    domino = DominoSlicer(
        y_log_likelihood_weight=0,
        y_hat_log_likelihood_weight=weight,
        n_mixture_components=n_slices,
        n_slices=n_slices,
        confusion_noise=0.001,
        random_state=42,
    )

    # Fit on validation data (no peeking at test)
    domino.fit(
        data=df_val, embeddings="clip(img)", targets="target", pred_probs="pred_probs"
    )

    # Predict on both val and test
    df_val["domino_slices"] = domino.predict(
        data=df_val, embeddings="clip(img)", targets="target", pred_probs="pred_probs"
    )
    df_test["domino_slices"] = domino.predict(
        data=df_test, embeddings="clip(img)", targets="target", pred_probs="pred_probs"
    )

    slice_preds_val = np.argmax(df_val["domino_slices"].data, axis=1)
    slice_preds_test = np.argmax(df_test["domino_slices"].data, axis=1)

    # Get predicted class labels from probabilities
    val_pred_labels = np.argmax(df_val["pred_probs"].data, axis=1)
    test_pred_labels = np.argmax(df_test["pred_probs"].data, axis=1)

    # Analysis on both val and test (now includes per-slice accuracy)
    print("\n=== Validation Set Analysis ===")
    val_res_df = analyze_slices(
        slice_preds_val,
        df_val["has_artifact"].data,
        df_val["target"].data,
        val_pred_labels,
    )
    val_res_df["split"] = "val"
    print(val_res_df)

    print("\n=== Test Set Analysis ===")
    test_res_df = analyze_slices(
        slice_preds_test,
        df_test["has_artifact"].data,
        df_test["target"].data,
        test_pred_labels,
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

        # Per-slice artifact rate bar chart
        artifact_data = [
            [f"Slice {row['slice']}", row["artifact_rate"]]
            for _, row in test_res_df.iterrows()
        ]
        artifact_table = wandb.Table(
            data=artifact_data, columns=["slice", "artifact_rate"]
        )
        wandb.log(
            {
                "test/slice_artifact_rate": wandb.plot.bar(
                    artifact_table,
                    "slice",
                    "artifact_rate",
                    title="Test Set: Per-Slice Artifact Rate",
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

    wandb.finish()


if __name__ == "__main__":
    main()
