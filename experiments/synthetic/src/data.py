import os
import requests
import tarfile
import shutil
import random
import hashlib
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from glob import glob


def download_imagenette(data_dir):
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, "imagenette2-320.tgz")

    if os.path.exists(os.path.join(data_dir, "imagenette2-320")):
        print("Imagenette already exists.")
        return

    print(f"Downloading Imagenette...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        with (
            open(tar_path, "wb") as f,
            tqdm(total=total_size, unit="iB", unit_scale=True) as pbar,
        ):
            for chunk in r.iter_content(chunk_size=8192):
                pbar.update(len(chunk))
                f.write(chunk)

    print("Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    os.remove(tar_path)


def add_vertical_line_artifact(image_path, output_path):
    """Add vertical hyperintense line artifact (center of image)."""
    img = cv2.imread(image_path)
    if img is None:
        return False
    h, w, _ = img.shape
    cv2.line(img, (w // 2, 0), (w // 2, h), (255, 255, 255), thickness=2)
    cv2.imwrite(output_path, img)
    return True


def add_hospital_tag_artifact(image_path, output_path):
    """Add hospital tag artifact (bottom-left corner rectangle with text)."""
    img = cv2.imread(image_path)
    if img is None:
        return False
    h, w, _ = img.shape
    # Draw white rectangle in bottom-left corner
    tag_h, tag_w = h // 6, w // 4
    cv2.rectangle(img, (0, h - tag_h), (tag_w, h), (255, 255, 255), -1)
    # Add "ID" text inside the tag
    font_scale = max(0.3, tag_h / 50)
    cv2.putText(
        img,
        "ID",
        (5, h - tag_h // 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        1,
    )
    cv2.imwrite(output_path, img)
    return True


ARTIFACT_FUNCTIONS = {
    "vertical_line": add_vertical_line_artifact,
    "hospital_tag": add_hospital_tag_artifact,
}


def stable_split(name, seed=42):
    """Deterministic split using MD5 hash (reproducible across runs/machines)."""
    h = hashlib.md5((name + str(seed)).encode()).hexdigest()
    return int(h, 16) % 2


def generate_synthetic_dataset(
    source_dir,
    target_dir,
    biased_classes=None,  # List of (class_idx, artifact_type) tuples
    prob_biased=0.95,
    prob_others=0.05,
    seed=42,
):
    """
    Generate synthetic dataset with artifact injection.

    Args:
        biased_classes: List of (class_idx, artifact_type) tuples.
                       Default: [(0, "vertical_line"), (1, "hospital_tag")]
                       artifact_type must be one of: "vertical_line", "hospital_tag"

    Fixed split logic:
    1. Assign final splits (train/val/test) FIRST using stable hash
    2. Apply artifact logic based on final split
    3. Guarantees reproducible artifact balance per split
    """
    if biased_classes is None:
        biased_classes = [(0, "vertical_line"), (1, "hospital_tag")]
    random.seed(seed)
    np.random.seed(seed)

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    # Create output directories
    for split in ["train", "val"]:
        os.makedirs(os.path.join(target_dir, split), exist_ok=True)

    # Phase 1: Collect all images and assign final splits FIRST
    metadata = []
    for orig_split in ["train", "val"]:
        source_split = os.path.join(source_dir, orig_split)
        classes = sorted(
            [
                d
                for d in os.listdir(source_split)
                if os.path.isdir(os.path.join(source_split, d))
            ]
        )

        for cls_idx, cls_name in enumerate(classes):
            src_cls_dir = os.path.join(source_split, cls_name)
            images = glob(os.path.join(src_cls_dir, "*.*"))

            for img_path in images:
                img_name = os.path.basename(img_path)

                # Determine final split using stable hash
                if orig_split == "train":
                    final_split = "train"
                else:
                    # Deterministic val/test split using MD5
                    final_split = "val" if stable_split(img_name, seed) == 0 else "test"

                metadata.append(
                    {
                        "src_path": img_path,
                        "cls_name": cls_name,
                        "cls_idx": cls_idx,
                        "img_name": img_name,
                        "final_split": final_split,
                    }
                )

    # Phase 2: Apply artifact logic based on FINAL split, then copy images
    final_metadata = []
    for item in tqdm(metadata, desc="Processing images"):
        cls_idx = item["cls_idx"]
        cls_name = item["cls_name"]
        img_name = item["img_name"]
        src_path = item["src_path"]
        final_split = item["final_split"]

        # Target directory: val and test both use "val" folder (images are the same)
        tgt_split_dir = os.path.join(
            target_dir, "val" if final_split != "train" else "train"
        )
        tgt_cls_dir = os.path.join(tgt_split_dir, cls_name)
        os.makedirs(tgt_cls_dir, exist_ok=True)
        tgt_path = os.path.join(tgt_cls_dir, img_name)

        # Determine which artifact type (if any) to apply
        biased_class_map = {
            cls_idx: artifact_type for cls_idx, artifact_type in biased_classes
        }
        artifact_types = [a[1] for a in biased_classes]

        has_artifact = 0
        artifact_type = None

        if final_split == "train":
            # Training: biased distribution (95% for biased classes, 5% for others)
            if cls_idx in biased_class_map:
                # This class is biased - use its assigned artifact
                if random.random() < prob_biased:
                    has_artifact = 1
                    artifact_type = biased_class_map[cls_idx]
            else:
                # Non-biased class - small chance of any artifact
                if random.random() < prob_others:
                    has_artifact = 1
                    artifact_type = random.choice(artifact_types)
        else:
            # Val/Test: decorrelated 50/50, random artifact type
            if random.random() < 0.5:
                has_artifact = 1
                artifact_type = random.choice(artifact_types)

        # Copy/modify image
        if has_artifact and artifact_type:
            artifact_fn = ARTIFACT_FUNCTIONS[artifact_type]
            artifact_fn(src_path, tgt_path)
        else:
            shutil.copy(src_path, tgt_path)

        final_metadata.append(
            {
                "image_path": os.path.join(
                    "val" if final_split != "train" else "train", cls_name, img_name
                ),
                "target": cls_idx,
                "has_artifact": has_artifact,
                "artifact_type": artifact_type if has_artifact else None,
                "split": final_split,
            }
        )

    # Create dataframes for each split
    df = pd.DataFrame(final_metadata)
    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()

    # Save CSVs
    df_train.to_csv(os.path.join(target_dir, "train.csv"), index=False)
    df_val.to_csv(os.path.join(target_dir, "val.csv"), index=False)
    df_test.to_csv(os.path.join(target_dir, "test.csv"), index=False)

    # Print statistics
    print(f"\nDataset statistics:")
    print(
        f"  Train: {len(df_train)} images, artifact rate: {df_train['has_artifact'].mean():.3f}"
    )
    print(
        f"  Val:   {len(df_val)} images, artifact rate: {df_val['has_artifact'].mean():.3f}"
    )
    print(
        f"  Test:  {len(df_test)} images, artifact rate: {df_test['has_artifact'].mean():.3f}"
    )

    return df_train, df_val, df_test
