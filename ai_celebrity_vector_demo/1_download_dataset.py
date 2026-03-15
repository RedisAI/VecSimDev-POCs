#!/usr/bin/env python3
"""
Download the Pins Face Recognition Dataset from Kaggle.

This script downloads the celebrity face dataset containing 105 celebrities
with approximately 17,500 images total.

Dataset: https://www.kaggle.com/datasets/hereisburak/pins-face-recognition
Celebrities include: Tom Holland, Zendaya, Elon Musk, Taylor Swift, RDJ, Chris Evans, etc.

Requirements:
- Kaggle API credentials configured (~/.kaggle/kaggle.json)
- Run: kaggle datasets download -d hereisburak/pins-face-recognition
"""

import os
import zipfile
import subprocess
import sys
from pathlib import Path

# Configuration
DATASET_NAME = "hereisburak/pins-face-recognition"
DATASET_DIR = "pins_dataset"
ZIP_FILE = "pins-face-recognition.zip"


def check_kaggle_credentials():
    """Check if Kaggle API credentials are configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("❌ Kaggle API credentials not found!")
        print("\nTo set up Kaggle API:")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Scroll to 'API' section and click 'Create New Token'")
        print("3. This downloads kaggle.json")
        print("4. Move it to ~/.kaggle/kaggle.json")
        print("5. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    return True


def download_dataset():
    """Download the Pins celebrity dataset from Kaggle."""
    print("🔄 Downloading Pins Face Recognition Dataset from Kaggle...")
    print(f"   Dataset: {DATASET_NAME}")
    
    try:
        # Download using Kaggle API
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", DATASET_NAME],
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ Download complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ Kaggle CLI not found. Install it with: pip install kaggle")
        return False


def extract_dataset():
    """Extract the downloaded zip file."""
    if not os.path.exists(ZIP_FILE):
        print(f"❌ Zip file not found: {ZIP_FILE}")
        return False
    
    print(f"📦 Extracting dataset to {DATASET_DIR}/...")
    
    try:
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(DATASET_DIR)
        print("✅ Extraction complete!")
        
        # Remove zip file to save space
        os.remove(ZIP_FILE)
        print(f"🗑️  Removed {ZIP_FILE}")
        
        return True
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False


def verify_dataset():
    """Verify the dataset structure and count images."""
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Dataset directory not found: {DATASET_DIR}")
        return False
    
    print("\n📊 Dataset Statistics:")
    
    # Count celebrities and images
    celebrity_dirs = [d for d in os.listdir(DATASET_DIR) 
                     if os.path.isdir(os.path.join(DATASET_DIR, d))]
    
    total_images = 0
    for celeb_dir in celebrity_dirs:
        celeb_path = os.path.join(DATASET_DIR, celeb_dir)
        images = [f for f in os.listdir(celeb_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_images += len(images)
    
    print(f"   Celebrities: {len(celebrity_dirs)}")
    print(f"   Total Images: {total_images}")
    print(f"   Average per Celebrity: {total_images // len(celebrity_dirs) if celebrity_dirs else 0}")
    
    # Show sample celebrities
    print(f"\n   Sample Celebrities: {', '.join(celebrity_dirs[:10])}")
    
    return True


def main():
    """Main execution function."""
    print("=" * 60)
    print("  PINS CELEBRITY DATASET DOWNLOADER")
    print("=" * 60)
    
    # Check if dataset already exists
    if os.path.exists(DATASET_DIR):
        response = input(f"\n⚠️  Dataset directory '{DATASET_DIR}' already exists. Re-download? (y/n): ")
        if response.lower() != 'y':
            print("✅ Using existing dataset.")
            verify_dataset()
            return
    
    # Check Kaggle credentials
    if not check_kaggle_credentials():
        sys.exit(1)
    
    # Download dataset
    if not download_dataset():
        sys.exit(1)
    
    # Extract dataset
    if not extract_dataset():
        sys.exit(1)
    
    # Verify dataset
    verify_dataset()
    
    print("\n" + "=" * 60)
    print("✅ Dataset ready! You can now run: python 2_generate_embeddings.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

