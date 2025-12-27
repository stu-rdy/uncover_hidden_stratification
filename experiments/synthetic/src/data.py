import os
import requests
import tarfile
import shutil
import random
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

def download_imagenette(data_dir):
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, "imagenette2-160.tgz")
    
    if os.path.exists(os.path.join(data_dir, "imagenette2-160")):
        print("Imagenette already exists.")
        return

    print(f"Downloading Imagenette...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(tar_path, 'wb') as f, tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                pbar.update(len(chunk))
                f.write(chunk)
    
    print("Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    os.remove(tar_path)

def add_vertical_line_artifact(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        return False
    h, w, _ = img.shape
    cv2.line(img, (w//2, 0), (w//2, h), (255, 255, 255), thickness=2)
    cv2.imwrite(output_path, img)
    return True

def generate_synthetic_dataset(source_dir, target_dir, biased_class_idx=0, 
                               prob_biased=0.95, prob_others=0.05, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    metadata = []
    
    for split in ["train", "val"]:
        source_split = os.path.join(source_dir, split)
        target_split = os.path.join(target_dir, split)
        os.makedirs(target_split)
        
        classes = sorted([d for d in os.listdir(source_split) if os.path.isdir(os.path.join(source_split, d))])
        
        for cls_idx, cls_name in enumerate(tqdm(classes, desc=f"Generating {split}")):
            src_cls_dir = os.path.join(source_split, cls_name)
            tgt_cls_dir = os.path.join(target_split, cls_name)
            os.makedirs(tgt_cls_dir)
            
            images = glob(os.path.join(src_cls_dir, "*.*"))
            for img_path in images:
                img_name = os.path.basename(img_path)
                tgt_path = os.path.join(tgt_cls_dir, img_name)
                
                # Logic for artifact
                has_artifact = 0
                if split == "train":
                    p = prob_biased if cls_idx == biased_class_idx else prob_others
                    if random.random() < p:
                        has_artifact = 1
                else: # 50/50 for evaluation
                    if random.random() < 0.5:
                        has_artifact = 1
                
                if has_artifact:
                    add_vertical_line_artifact(img_path, tgt_path)
                else:
                    shutil.copy(img_path, tgt_path)
                
                metadata.append({
                    "image_path": os.path.join(split, cls_name, img_name),
                    "target": cls_idx,
                    "has_artifact": has_artifact,
                    "split": split if split == "train" else ("val" if random.random() < 0.5 else "test")
                })

    df = pd.DataFrame(metadata)
    # Split val into val and test if not already handled by logic above
    # Re-doing split for clarity
    df_train = df[df['split'] == 'train']
    val_test = df[df['split'] != 'train']
    
    # Actually the split logic in line 84-85 is a bit random, let's fix it
    df_val = val_test.sample(frac=0.5, random_state=seed)
    df_test = val_test.drop(df_val.index)
    
    df_val.loc[:, 'split'] = 'val'
    df_test.loc[:, 'split'] = 'test'
    
    df_train.to_csv(os.path.join(target_dir, "train.csv"), index=False)
    df_val.to_csv(os.path.join(target_dir, "val.csv"), index=False)
    df_test.to_csv(os.path.join(target_dir, "test.csv"), index=False)
    
    return df_train, df_val, df_test
