import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import wandb
from tqdm.auto import tqdm

# Add project root and experiment src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../"))
sys.path.append(PROJECT_ROOT)

from experiments.synthetic.src.model import get_model, train_one_epoch, evaluate
from experiments.synthetic.src.data import (
    generate_synthetic_dataset,
)  # Just for reference if needed
from src.data_loader import CSVDatasetWithName


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--project", default="synthetic_imagenette")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Load YAML config if provided
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    # Merge args with config (args override config)
    epochs = (
        args.epochs
        if not config.get("training")
        else config["training"].get("epochs", args.epochs)
    )
    lr = (
        args.lr
        if not config.get("training")
        else config["training"].get("learning_rate", args.lr)
    )
    bs = (
        args.bs if not config.get("data") else config["data"].get("batch_size", args.bs)
    )

    # WandB Init
    wb_conf = config.get("wandb", {})
    project = wb_conf.get("project", args.project)

    # Device logic
    if args.device:
        device = torch.device(args.device)
    elif config.get("training", {}).get("device"):
        device = torch.device(config["training"]["device"])
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print(f"Using device: {device}")

    # Model and Data params from config
    arch = config.get("model", {}).get("arch", "resnet50")
    num_workers = config.get("data", {}).get("num_workers", 2)

    # Data
    data_root = os.path.join(PROJECT_ROOT, "data/synthetic_imagenette")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_set = CSVDatasetWithName(
        data_root,
        os.path.join(data_root, "train.csv"),
        "image_path",
        "target",
        "has_artifact",
        transform=transform,
        verbose=False,
    )
    val_set = CSVDatasetWithName(
        data_root,
        os.path.join(data_root, "val.csv"),
        "image_path",
        "target",
        "has_artifact",
        transform=transform,
        verbose=False,
    )
    test_set = CSVDatasetWithName(
        data_root,
        os.path.join(data_root, "test.csv"),
        "image_path",
        "target",
        "has_artifact",
        transform=transform,
        verbose=False,
    )

    train_loader = DataLoader(
        train_set, batch_size=bs, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(val_set, batch_size=bs, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=bs, num_workers=num_workers)

    # Model
    model = get_model(arch=arch, num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    wandb.init(
        project=project,
        entity=wb_conf.get("entity"),
        tags=wb_conf.get("tags"),
        notes=wb_conf.get("notes"),
        config=config or args,
    )

    best_acc = 0
    save_dir = os.path.join(PROJECT_ROOT, "saves/synthetic")
    os.makedirs(save_dir, exist_ok=True)

    # Early Stopping Setup
    es_config = config.get("training", {}).get("early_stopping", {})
    early_stopping_enabled = es_config.get("enabled", False)
    patience = es_config.get("patience", 5)
    # Default to val_acc (stable signal). Avoid worst_group_acc for stopping.
    monitor_metric = es_config.get("monitor", "val_acc")

    epochs_no_improve = 0
    best_monitor_val = -float("inf")  # Safe for any metric range

    for epoch in tqdm(range(epochs), desc="Epochs", ascii=True, mininterval=1.0):
        train_results = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_results = evaluate(model, val_loader, device)

        # Base logs
        log_dict = {
            "loss": train_results["loss"],
            "train_acc": train_results["acc"],
            "val_acc": val_results["acc"],
            "epoch": epoch,
        }

        # Detailed metrics if configured
        metrics_config = config.get("metrics", {})
        if metrics_config.get("log_worst_group"):
            detailed_metrics = compute_detailed_metrics(
                val_results, metrics_config, split="val"
            )
            log_dict.update(detailed_metrics)

        # Training distribution metrics (shows training was biased)
        if metrics_config.get("log_training_metrics"):
            train_dist = compute_training_distribution_metrics(train_results)
            log_dict.update(train_dist)

        tqdm.write(
            f"Epoch {epoch}: Loss {train_results['loss']:.4f}, Train Acc {train_results['acc']:.4f}, Val Acc {val_results['acc']:.4f}"
        )

        # Early Stopping Logic
        if early_stopping_enabled:
            # Normalize metric key (handle both "val_acc" and "worst_group_acc" formats)
            if "/" not in monitor_metric:
                monitor_key = (
                    f"val/{monitor_metric}"
                    if monitor_metric != "val_acc"
                    else monitor_metric
                )
            else:
                monitor_key = monitor_metric

            current_monitor_val = log_dict.get(monitor_key)
            if current_monitor_val is None:
                raise ValueError(
                    f"Early stopping metric '{monitor_key}' not found in log_dict. "
                    f"Available keys: {list(log_dict.keys())}"
                )

            if current_monitor_val > best_monitor_val:
                best_monitor_val = current_monitor_val
                epochs_no_improve = 0
                tqdm.write(
                    f"  --> Best {monitor_metric} improved to {best_monitor_val:.4f}. Saving model."
                )
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            else:
                epochs_no_improve += 1
                tqdm.write(
                    f"  --> {monitor_metric} did not improve. Patience: {epochs_no_improve}/{patience}"
                )

            if epochs_no_improve >= patience:
                tqdm.write(f"Early stopping triggered at epoch {epoch}!")
                wandb.log({"early_stop_epoch": epoch})
                break
        else:
            # Standard best model saving if ES is disabled
            if val_results["acc"] > best_acc:
                best_acc = val_results["acc"]
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

        wandb.log(log_dict)

    print("Training complete. Evaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth")))
    test_results = evaluate(model, test_loader, device)

    test_log = {"test/acc": test_results["acc"]}
    if metrics_config:
        test_log.update(
            compute_detailed_metrics(test_results, metrics_config, split="test")
        )
    wandb.log(test_log)

    print(f"Test Acc: {test_results['acc']:.4f}")

    # Save predictions for both val and test (analysis script needs both)
    import pandas as pd

    # Val predictions (Domino fits on val set)
    val_results_final = evaluate(model, val_loader, device)
    df_val_preds = pd.DataFrame(
        val_results_final["probs"], columns=[f"pred_{i}" for i in range(10)]
    )
    df_val_preds["name"] = val_results_final["names"]
    df_val_preds["gt"] = val_results_final["labels"]
    df_val_preds["group"] = val_results_final["groups"]
    df_val_preds.to_csv(os.path.join(save_dir, "val_predictions.csv"), index=False)
    print(f"Saved predictions to {os.path.join(save_dir, 'val_predictions.csv')}")

    # Test predictions
    df_preds = pd.DataFrame(
        test_results["probs"], columns=[f"pred_{i}" for i in range(10)]
    )
    df_preds["name"] = test_results["names"]
    df_preds["gt"] = test_results["labels"]
    df_preds["group"] = test_results["groups"]
    df_preds.to_csv(os.path.join(save_dir, "test_predictions.csv"), index=False)
    print(f"Saved predictions to {os.path.join(save_dir, 'test_predictions.csv')}")


def compute_detailed_metrics(results, metrics_config, split="val"):
    """Compute only essential metrics: worst_group_acc and confusion_matrix."""
    preds = results["preds"]
    labels = results["labels"]
    groups = results["groups"]  # has_artifact 0 or 1

    metrics = {}

    # Worst-group Accuracy (key metric for subgroup performance)
    if metrics_config.get("log_worst_group"):
        subgroup_accs = []
        for c in np.unique(labels):
            for g in np.unique(groups):
                mask = (labels == c) & (groups == g)
                if np.sum(mask) > 0:
                    acc = (preds[mask] == labels[mask]).mean()
                    subgroup_accs.append(acc)
        if subgroup_accs:
            metrics[f"{split}/worst_group_acc"] = min(subgroup_accs)

    # Confusion Matrix (only for test set)
    if metrics_config.get("log_confusion_matrix") and split == "test":
        metrics[f"{split}/confusion_matrix"] = wandb.plot.confusion_matrix(
            probs=None,
            y_true=labels,
            preds=preds,
            class_names=[str(i) for i in range(10)],
        )

    return metrics


def compute_training_distribution_metrics(results, biased_class_idx=0):
    """
    Compute training data distribution metrics to show class-conditional bias.
    Logs per-class artifact rates and artifact-class correlation.
    """
    labels = results["labels"]
    groups = results["groups"]  # 0 = no artifact, 1 = artifact

    metrics = {}

    # Overall artifact prevalence
    metrics["train/data/artifact_rate"] = groups.mean()

    # Per-class artifact rates (shows class-conditional bias explicitly)
    # Expect ~0.95 for biased class, ~0.05 for others
    for c in np.unique(labels):
        mask = labels == c
        metrics[f"train/data/artifact_rate/class_{int(c)}"] = groups[mask].mean()

    # Artifact–class correlation (summary of shortcut strength)
    biased_class_mask = labels == biased_class_idx
    if len(np.unique(groups)) > 1:
        metrics["train/data/artifact_class_correlation"] = np.corrcoef(
            groups, biased_class_mask.astype(int)
        )[0, 1]

    return metrics


if __name__ == "__main__":
    main()
