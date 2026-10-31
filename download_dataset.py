"""
Automated Dataset Downloader for Image Colorization
Downloads the COCO mini dataset (2,000 train images, 200 val images).
Ideal for 15-20 minutes training on an RTX 4050 GPU.
"""

import os
import zipfile
import requests
from tqdm import tqdm
import shutil

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
ZIP_URL = "http://images.cocodataset.org/zips/val2017.zip"
ZIP_PATH = os.path.join(DATA_DIR, "val2017.zip")

NUM_TRAIN = 2000
NUM_VAL = 200

def download_file(url, target_path):
    print(f"Downloading dataset from: {url}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024 * 1024  # 1 MB

    with open(target_path, "wb") as f, tqdm(
        total=total_size, unit="iB", unit_scale=True, desc="Downloading"
    ) as bar:
        for chunk in response.iter_content(block_size):
            f.write(chunk)
            bar.update(len(chunk))

def prepare_dataset():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    if not os.path.exists(ZIP_PATH):
        download_file(ZIP_URL, ZIP_PATH)

    print("\nExtracting and organizing images...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        # Get list of images
        img_members = [m for m in z.namelist() if m.endswith((".jpg", ".png", ".jpeg"))]
        
        print(f"Found {len(img_members)} total images.")
        
        # Extract train images
        print(f"Extracting {NUM_TRAIN} images to data/raw/train...")
        for i, member in enumerate(tqdm(img_members[:NUM_TRAIN], desc="Train set")):
            filename = os.path.basename(member)
            target = os.path.join(TRAIN_DIR, filename)
            with z.open(member) as source, open(target, "wb") as dest:
                shutil.copyfileobj(source, dest)
                
        # Extract val images
        print(f"Extracting {NUM_VAL} images to data/raw/val...")
        for member in tqdm(img_members[NUM_TRAIN:NUM_TRAIN + NUM_VAL], desc="Val set"):
            filename = os.path.basename(member)
            target = os.path.join(VAL_DIR, filename)
            with z.open(member) as source, open(target, "wb") as dest:
                shutil.copyfileobj(source, dest)

    # Clean up zip file to save disk space
    if os.path.exists(ZIP_PATH):
        print(f"\nCleaning up {ZIP_PATH} to save disk space...")
        os.remove(ZIP_PATH)

    print("\n" + "="*50)
    print(f"SUCCESS: Dataset ready for 15-20 min training!")
    print(f"Train images: {len(os.listdir(TRAIN_DIR))} in data/raw/train")
    print(f"Val images:   {len(os.listdir(VAL_DIR))} in data/raw/val")
    print("="*50)

if __name__ == "__main__":
    prepare_dataset()
