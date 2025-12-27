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
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--predictions_csv', type=str, default=os.path.join(PROJECT_ROOT, "saves/synthetic/test_predictions.csv"))
    parser.add_argument('--n_slices', type=int, default=10)
    parser.add_argument('--weight', type=float, default=10.0)
    parser.add_argument('--project', default="synthetic_imagenette")
    args = parser.parse_args()

    # Load YAML config if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Merge args with config
    n_slices = args.n_slices if not config.get('analysis') else config['analysis'].get('n_slices', args.n_slices)
    weight = args.weight if not config.get('analysis') else config['analysis'].get('weight', args.weight)
    
    # WandB Init
    wb_conf = config.get('wandb', {})
    project = wb_conf.get('project', args.project)

    wandb.init(
        project=project,
        entity=wb_conf.get('entity'),
        tags=wb_conf.get('tags'),
        notes=wb_conf.get('notes'),
        job_type="analysis", 
        config=config or args
    )

    data_root = os.path.join(PROJECT_ROOT, "data/synthetic_imagenette")
    embed_path = os.path.join(data_root, "test_embeddings.mk")
    
    if not os.path.exists(embed_path):
        print(f"Embeddings not found at {embed_path}. Run 4_extract_features.py first.")
        wandb.finish()
        return

    print("Loading data...")
    df_test = mk.read(embed_path)
    preds_df = pd.read_csv(args.predictions_csv)
    
    # Merge predictions
    pred_cols = [c for c in preds_df.columns if c.startswith('pred_')]
    sorted_pred_cols = sorted(pred_cols, key=lambda x: int(x.split('_')[1]))
    probs = preds_df[sorted_pred_cols].values
    df_test["pred_probs"] = probs

    # Run Domino
    print(f"Running Domino with {n_slices} slices and weight {weight}...")
    domino = DominoSlicer(
        y_log_likelihood_weight=0,
        y_hat_log_likelihood_weight=weight,
        n_mixture_components=n_slices,
        n_slices=n_slices,
        confusion_noise=0.001,
        random_state=42
    )
    
    domino.fit(
        data=df_test,
        embeddings="clip(img)",
        targets="target",
        pred_probs="pred_probs"
    )
    
    df_test["domino_slices"] = domino.predict(
        data=df_test,
        embeddings="clip(img)",
        targets="target",
        pred_probs="pred_probs"
    )
    
    slice_preds = np.argmax(df_test["domino_slices"].data, axis=1)
    
    # Analysis
    res_df = analyze_slices(slice_preds, df_test["has_artifact"].data, df_test["target"].data)
    
    print("\nSlice Analysis Summary:")
    print(res_df)
    
    out_path = os.path.join(PROJECT_ROOT, "results/synthetic_analysis.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    res_df.to_csv(out_path, index=False)
    print(f"\nSaved detailed analysis to {out_path}")
    
    # Log results to wandb
    table = wandb.Table(dataframe=res_df)
    wandb.log({"analysis_results": table})
    wandb.finish()

if __name__ == "__main__":
    main()
