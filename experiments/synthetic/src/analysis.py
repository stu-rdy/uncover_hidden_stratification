import numpy as np
import pandas as pd
from domino import DominoSlicer


def run_subgroup_discovery(
    embeddings, targets, pred_probs, n_slices=10, weight=10.0, seed=42
):
    """
    embeddings: Meerkat column or numpy array of embeddings (N, D)
    targets: array of ground truth labels (N,)
    pred_probs: array of prediction probabilities (N, C)
    """
    # Note: This assumes embeddings is already in the metadata/dataframe if using DominoSlicer.fit(data=df)
    # For modularity, we might need a simpler wrapper if we want to pass raw arrays.
    # Domino usually works best with Meerkat DataFrames.
    pass


def analyze_slices(slice_assignments, has_artifact, targets, predictions=None):
    """
    Analyze discovered slices.

    Args:
        slice_assignments: array of slice assignments (N,)
        has_artifact: array of artifact presence (N,)
        targets: array of ground truth labels (N,)
        predictions: optional array of predicted labels (N,) for computing accuracy

    Returns:
        DataFrame with per-slice statistics
    """
    df = pd.DataFrame(
        {"slice": slice_assignments, "has_artifact": has_artifact, "target": targets}
    )

    if predictions is not None:
        df["prediction"] = predictions

    unique_slices = np.unique(slice_assignments)
    results = []

    for sl in unique_slices:
        subset = df[df["slice"] == sl]
        size = len(subset)
        artifact_rate = subset["has_artifact"].mean()
        dom_class = subset["target"].mode()[0]
        dom_class_perc = (subset["target"] == dom_class).mean()

        result = {
            "slice": int(sl),
            "size": size,
            "artifact_rate": float(artifact_rate),
            "dom_class": int(dom_class),
            "dom_class_perc": float(dom_class_perc),
        }

        # Add accuracy if predictions are provided
        if predictions is not None:
            accuracy = (subset["prediction"] == subset["target"]).mean()
            result["accuracy"] = float(accuracy)

        results.append(result)

    return pd.DataFrame(results)
