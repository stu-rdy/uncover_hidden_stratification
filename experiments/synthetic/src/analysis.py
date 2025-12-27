import numpy as np
import pandas as pd
from domino import DominoSlicer

def run_subgroup_discovery(embeddings, targets, pred_probs, n_slices=10, weight=10.0, seed=42):
    """
    embeddings: Meerkat column or numpy array of embeddings (N, D)
    targets: array of ground truth labels (N,)
    pred_probs: array of prediction probabilities (N, C)
    """
    # Note: This assumes embeddings is already in the metadata/dataframe if using DominoSlicer.fit(data=df)
    # For modularity, we might need a simpler wrapper if we want to pass raw arrays.
    # Domino usually works best with Meerkat DataFrames.
    pass

def analyze_slices(slice_assignments, has_artifact, targets):
    df = pd.DataFrame({
        'slice': slice_assignments,
        'has_artifact': has_artifact,
        'target': targets
    })
    
    unique_slices = np.unique(slice_assignments)
    results = []
    
    for sl in unique_slices:
        subset = df[df['slice'] == sl]
        size = len(subset)
        artifact_rate = subset['has_artifact'].mean()
        dom_class = subset['target'].mode()[0]
        dom_class_perc = (subset['target'] == dom_class).mean()
        
        results.append({
            "slice": int(sl),
            "size": size,
            "artifact_rate": float(artifact_rate),
            "dom_class": int(dom_class),
            "dom_class_perc": float(dom_class_perc)
        })
        
    return pd.DataFrame(results)
