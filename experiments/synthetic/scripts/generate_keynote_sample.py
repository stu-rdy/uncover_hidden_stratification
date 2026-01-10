import os
import cv2
import numpy as np
import sys

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../"))
sys.path.append(PROJECT_ROOT)

from experiments.synthetic.src.data import apply_artifact_to_array


def generate_sample(output_path="keynote_sample.png"):
    # Create a dummy image (colored background)
    h, w = 320, 320
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :160] = [100, 150, 200]  # Color 1
    img[:, 160:] = [200, 100, 150]  # Color 2

    # 1. Apply heavy blur and noise (mocking generation pipeline)
    img = cv2.GaussianBlur(img, (0, 0), sigmaX=3.0)
    noise = np.random.normal(0, 0.05 * 255, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 2. Convert to Grayscale (X-ray style)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 3. Apply artifact (remains sharp and high contrast)
    img = apply_artifact_to_array(img, "hospital_tag")

    # Save image
    cv2.imwrite(output_path, img)
    print(f"Sample generated at {output_path}")


if __name__ == "__main__":
    generate_sample()
