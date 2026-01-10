import os
import sys
import argparse
import yaml

# Add project root and experiment src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../"))
sys.path.append(PROJECT_ROOT)

from experiments.synthetic.src.data import generate_synthetic_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--project", default="synthetic_imagenette")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true", help="Skip wandb logging")
    args = parser.parse_args()

    # Load YAML config if provided
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    # Optional WandB Init
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb

        wb_conf = config.get("wandb", {})
        project = wb_conf.get("project", args.project)
        wandb.init(
            project=project,
            entity=wb_conf.get("entity"),
            tags=wb_conf.get("tags"),
            notes=wb_conf.get("notes"),
            job_type="data_generation",
            config=config or args,
        )

    seed = config.get("experiment", {}).get("seed", args.seed)

    # Synthetic artifact settings from config
    syn_conf = config.get("synthetic", {})
    biased_class_idx = syn_conf.get("biased_class_idx", 0)
    hidden_artifact = syn_conf.get("hidden_artifact", "hospital_tag")
    known_artifact = syn_conf.get("known_artifact", "vertical_line")
    prob_hidden_biased = syn_conf.get("prob_hidden_biased", 0.8)
    prob_hidden_others = syn_conf.get("prob_hidden_others", 0.05)
    prob_known = syn_conf.get("prob_known", 0.5)
    blur_sigma = syn_conf.get("blur_sigma", 0.0)
    noise_std = syn_conf.get("noise_std", 0.0)
    grayscale = syn_conf.get("grayscale", False)

    source_dir = os.path.join(PROJECT_ROOT, "data/imagenette2-320")
    target_dir = os.path.join(PROJECT_ROOT, "data/synthetic_imagenette")

    if not os.path.exists(source_dir):
        print("Source data not found. Run 1_setup_data.py first.")
        if use_wandb:
            wandb.finish()
        return

    df_train, df_val, df_test = generate_synthetic_dataset(
        source_dir,
        target_dir,
        biased_class_idx=biased_class_idx,
        hidden_artifact=hidden_artifact,
        known_artifact=known_artifact,
        prob_hidden_biased=prob_hidden_biased,
        prob_hidden_others=prob_hidden_others,
        prob_known=prob_known,
        blur_sigma=blur_sigma,
        noise_std=noise_std,
        grayscale=grayscale,
        seed=seed,
    )

    # Log stats
    for split, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        artifact_rate = df["has_artifact"].mean()
        print(f"  {split}: {len(df)} images, artifact rate: {artifact_rate:.3f}")
        if use_wandb:
            import wandb

            wandb.log(
                {f"{split}/size": len(df), f"{split}/artifact_rate": artifact_rate}
            )

    print("Synthetic dataset generation complete.")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
