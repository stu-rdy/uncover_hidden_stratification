import os
import sys

# Add project root and experiment src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../"))
sys.path.append(PROJECT_ROOT)

from experiments.synthetic.src.data import download_imagenette

def main():
    data_dir = os.path.join(PROJECT_ROOT, "data")
    download_imagenette(data_dir)
    print("Data setup complete.")

if __name__ == "__main__":
    main()
