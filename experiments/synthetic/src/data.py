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
    # Dominant artifact: thicker, high-contrast line
    cv2.line(img, (w // 2, 0), (w // 2, h), (255, 255, 255), thickness=5)
    cv2.imwrite(output_path, img)
    return True


def add_hospital_tag_artifact(image_path, output_path):
    """Add hospital tag artifact (bottom-left corner rectangle with text)."""
    img = cv2.imread(image_path)
    if img is None:
        return False
    h, w, _ = img.shape
    # Dominant artifact: larger, fixed rectangle in bottom-left
    tag_h, tag_w = h // 4, w // 3
    cv2.rectangle(img, (0, h - tag_h), (tag_w, h), (255, 255, 255), -1)
    # Add text inside the tag with dynamic scaling to fit
    text = "ID - TAG"
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    # 1. Start with a default font scale
    font_scale = max(0.5, tag_h / 40)

    # 2. Adjust font scale to fit within tag_w (with 10% padding on each side)
    target_w = tag_w * 0.8
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    if text_w > target_w:
        font_scale *= target_w / text_w
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # 3. Calculate centered position
    text_x = int((tag_w - text_w) / 2)
    text_y = int((h - tag_h) + (tag_h + text_h) / 2)

    cv2.putText(
        img,
        text,
        (text_x, text_y),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
    )
    cv2.imwrite(output_path, img)
    return True


ARTIFACT_FUNCTIONS = {
    "vertical_line": add_vertical_line_artifact,
    "hospital_tag": add_hospital_tag_artifact,
}


def apply_artifact_to_array(img, artifact_type):
    """Apply artifact to image array (allows stacking multiple artifacts)."""
    h, w, _ = img.shape
    if artifact_type == "vertical_line":
        cv2.line(img, (w // 2, 0), (w // 2, h), (255, 255, 255), thickness=5)
    elif artifact_type == "hospital_tag":
        tag_h, tag_w = h // 4, w // 3
        cv2.rectangle(img, (0, h - tag_h), (tag_w, h), (255, 255, 255), -1)

        text = "ID - TAG"
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2

        # Adjust scale to fit
        font_scale = max(0.5, tag_h / 40)
        target_w = tag_w * 0.8
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        if text_w > target_w:
            font_scale *= target_w / text_w
            (text_w, text_h), baseline = cv2.getTextSize(
                text, font, font_scale, thickness
            )

        text_x = int((tag_w - text_w) / 2)
        text_y = int((h - tag_h) + (tag_h + text_h) / 2)

        cv2.putText(
            img,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
        )
    return img


def stable_split(name, seed=42):
    """Deterministic split using MD5 hash (reproducible across runs/machines)."""
    h = hashlib.md5((name + str(seed)).encode()).hexdigest()
    return int(h, 16) % 2


def generate_synthetic_dataset(
    source_dir,
    target_dir,
    biased_class_idx=0,
    hidden_artifact="hospital_tag",  # Hidden stratification (biased)
    known_artifact="vertical_line",  # Known attribute (independent)
    prob_hidden_biased=0.8,  # p for biased class (Bissoto uses 0.6, 0.7, 0.8)
    prob_hidden_others=0.05,  # p for non-biased classes
    prob_known: float = 0.5,  # Known artifact rate (independent of class)
    blur_sigma: float = 0.0,  # Gaussian blur applied to base image
    noise_std: float = 0.0,  # Gaussian noise applied to base image
    seed=42,
):
    """
    Generate synthetic dataset matching Bissoto et al. methodology.

    Two artifact types with different roles:
    - hidden_artifact (hospital_tag): HIDDEN stratification - biased toward one class
    - known_artifact (vertical_line): KNOWN attribute - independent of class, tracked as metadata

    Args:
        biased_class_idx: Class index that receives hidden artifact with high probability
        hidden_artifact: Artifact type for hidden stratification (default: hospital_tag)
        known_artifact: Artifact type for known attribute (default: vertical_line)
        prob_hidden_biased: Probability of hidden artifact for biased class (0.6, 0.7, 0.8)
        prob_hidden_others: Probability of hidden artifact for non-biased classes
        prob_known: Probability of known artifact (independent of class)
        blur_sigma: Sigma for Gaussian blur (destroys fine features)
        noise_std: Std dev for Gaussian noise (adds visual complexity)
        seed: Random seed for reproducibility
    """
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

        # Target directory: val and test both use "val" folder
        tgt_split_dir = os.path.join(
            target_dir, "val" if final_split != "train" else "train"
        )
        tgt_cls_dir = os.path.join(tgt_split_dir, cls_name)
        os.makedirs(tgt_cls_dir, exist_ok=True)
        tgt_path = os.path.join(tgt_cls_dir, img_name)

        # Determine artifacts
        has_hidden = 0  # Hospital tag (hidden stratification)
        has_known = 0  # Vertical line (known attribute)

        if final_split == "train":
            # Hidden artifact: biased distribution
            p_hidden = (
                prob_hidden_biased
                if cls_idx == biased_class_idx
                else prob_hidden_others
            )
            if random.random() < p_hidden:
                has_hidden = 1
            # Known artifact: independent of class
            if random.random() < prob_known:
                has_known = 1
        else:
            # Val/Test: decorrelated 50/50 for both artifacts
            if random.random() < 0.5:
                has_hidden = 1
            if random.random() < 0.5:
                has_known = 1

        # Apply artifacts to image (can have both, one, or none)
        img = cv2.imread(src_path)
        if img is not None:
            # Apply Noise and Blur to the BASE image first to reduce base signal
            if blur_sigma > 0:
                img = cv2.GaussianBlur(img, (0, 0), sigmaX=blur_sigma)

            if noise_std > 0:
                noise = np.random.normal(0, noise_std * 255, img.shape).astype(
                    np.float32
                )
                img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            if has_hidden:
                img = apply_artifact_to_array(img, hidden_artifact)
            if has_known:
                img = apply_artifact_to_array(img, known_artifact)
            cv2.imwrite(tgt_path, img)
        else:
            shutil.copy(src_path, tgt_path)

        final_metadata.append(
            {
                "image_path": os.path.join(
                    "val" if final_split != "train" else "train", cls_name, img_name
                ),
                "target": cls_idx,
                "has_artifact": has_hidden,  # Primary artifact for analysis (hidden)
                "has_hidden_artifact": has_hidden,  # Hospital tag (hidden stratification)
                "has_known_artifact": has_known,  # Vertical line (known attribute)
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
