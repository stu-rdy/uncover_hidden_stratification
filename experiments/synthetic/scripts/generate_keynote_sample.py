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
    # Create a dummy image (gray background)
    h, w = 320, 320
    img = np.full((h, w, 3), 128, dtype=np.uint8)

    # Apply hospital tag artifact
    img = apply_artifact_to_array(img, "hospital_tag")

    # Save image
    cv2.imwrite(output_path, img)
    print(f"Sample generated at {output_path}")


if __name__ == "__main__":
    generate_sample()
