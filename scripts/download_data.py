"""
Dataset Download Helper Script.

Supports downloading the Customer Support on Twitter dataset from Kaggle
using the Kaggle API, with manual instructions fallback.
"""

import os
import sys
import argparse


def download_dataset(output_dir: str = "data/raw/twcs"):
    """
    Downloads the TWCS dataset from Kaggle via kaggle API CLI or python package.
    """
    os.makedirs(output_dir, exist_ok=True)
    target_csv = os.path.join(output_dir, "twcs.csv")

    if os.path.exists(target_csv):
        print(f"[INFO] Dataset already exists at {target_csv}")
        return

    print("==================================================")
    print("Kaggle Dataset: thoughtvector/customer-support-on-twitter")
    print("==================================================")

    try:
        import kaggle
        print("[INFO] Attempting download via Kaggle Python API...")
        kaggle.api.dataset_download_files("thoughtvector/customer-support-on-twitter", path=output_dir, unzip=True)
        print(f"[SUCCESS] Downloaded and unzipped to {output_dir}")
        return
    except Exception as e:
        print(f"[WARNING] Kaggle API automated download not available: {e}")

    print("\n--------------------------------------------------")
    print("MANUAL DOWNLOAD INSTRUCTIONS:")
    print("1. Visit: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter")
    print("2. Download the twcs.csv file.")
    print(f"3. Place twcs.csv into: {os.path.abspath(output_dir)}")
    print("--------------------------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download TWCS dataset from Kaggle")
    parser.add_argument("--output_dir", default="data/raw/twcs", help="Directory to save dataset")
    args = parser.parse_args()
    download_dataset(args.output_dir)
