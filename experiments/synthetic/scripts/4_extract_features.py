import os
import sys
import argparse
import yaml
import torch
import meerkat as mk
from domino import embed

# Add project root and experiment src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../"))
sys.path.append(PROJECT_ROOT)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--bs', type=int, default=128)
    args = parser.parse_args()

    # Load YAML config if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    bs = args.bs if not config.get('data') else config['data'].get('batch_size', args.bs)
    
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    print(f"Using device: {device}")
    data_root = os.path.join(PROJECT_ROOT, "data/synthetic_imagenette")

    for split in ["train", "val", "test"]:
        csv_path = os.path.join(data_root, f"{split}.csv")
        if not os.path.exists(csv_path):
            continue
            
        print(f"Extracting features for {split}...")
        df = mk.from_csv(csv_path)
        df["img"] = mk.image(df["image_path"], base_dir=data_root)
        
        df_embed = embed(
            df,
            input_col="img",
            encoder="clip",
            modality="image",
            device=device,
            batch_size=bs
        )
        
        out_path = os.path.join(data_root, f"{split}_embeddings.mk")
        df_embed.write(out_path)
        print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
