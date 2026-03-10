#!/usr/bin/env python3
"""
Generate face embeddings for the celebrity dataset.

This script uses the face_recognition library to generate 128-dimensional
face embeddings for all celebrity images. The embeddings are saved to a
pickle file for reuse.

The face_recognition library:
- FREE (no API costs)
- Runs locally
- Generates 128-dimensional vectors
- Based on dlib's ResNet model
"""

import os
import pickle
import face_recognition
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Configuration
DATASET_DIR = "pins_dataset/105_classes_pins_dataset"
OUTPUT_FILE = "celebrity_embeddings.pkl"
MAX_IMAGES_PER_CELEBRITY = None  # None = process all images


def load_image_safe(image_path):
    """Safely load an image and convert to RGB."""
    try:
        img = Image.open(image_path)
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)
    except Exception as e:
        print(f"   ⚠️  Error loading {image_path}: {e}")
        return None


def generate_embedding(image_path):
    """
    Generate a 128-dimensional face embedding for an image.
    
    Returns:
        numpy.ndarray: 128-dim embedding vector, or None if no face detected
    """
    try:
        # Load image
        image = load_image_safe(image_path)
        if image is None:
            return None
        
        # Detect faces and generate embeddings
        # face_encodings returns a list of 128-dim embeddings (one per face)
        encodings = face_recognition.face_encodings(image)
        
        if len(encodings) == 0:
            return None  # No face detected
        
        # Return the first face encoding (assuming one face per image)
        return encodings[0]
    
    except Exception as e:
        print(f"   ⚠️  Error processing {image_path}: {e}")
        return None


def process_dataset():
    """
    Process all celebrity images and generate embeddings.
    
    Returns:
        list: List of dicts with keys: 'name', 'image_path', 'embedding'
    """
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Dataset directory not found: {DATASET_DIR}")
        print("   Please run: python 1_download_dataset.py")
        return []
    
    embeddings_data = []
    
    # Get all celebrity directories
    celebrity_dirs = sorted([d for d in os.listdir(DATASET_DIR) 
                            if os.path.isdir(os.path.join(DATASET_DIR, d))])
    
    print(f"📊 Found {len(celebrity_dirs)} celebrities")
    print(f"🔄 Generating embeddings...\n")
    
    total_processed = 0
    total_failed = 0
    
    for celeb_name in celebrity_dirs:
        celeb_path = os.path.join(DATASET_DIR, celeb_name)
        
        # Get all image files
        image_files = sorted([f for f in os.listdir(celeb_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        # Limit images per celebrity if specified
        if MAX_IMAGES_PER_CELEBRITY:
            image_files = image_files[:MAX_IMAGES_PER_CELEBRITY]
        
        print(f"👤 {celeb_name}: Processing {len(image_files)} images...")
        
        celeb_success = 0
        celeb_failed = 0
        
        for img_file in tqdm(image_files, desc=f"   {celeb_name}", leave=False):
            img_path = os.path.join(celeb_path, img_file)
            
            # Generate embedding
            embedding = generate_embedding(img_path)
            
            if embedding is not None:
                embeddings_data.append({
                    'name': celeb_name,
                    'image_path': img_path,
                    'embedding': embedding
                })
                celeb_success += 1
            else:
                celeb_failed += 1
        
        total_processed += celeb_success
        total_failed += celeb_failed
        
        print(f"   ✅ Success: {celeb_success}, ❌ Failed: {celeb_failed}")
    
    print(f"\n📊 Summary:")
    print(f"   Total embeddings generated: {total_processed}")
    print(f"   Total failed: {total_failed}")
    if total_processed + total_failed > 0:
        print(f"   Success rate: {total_processed / (total_processed + total_failed) * 100:.1f}%")
    else:
        print(f"   Success rate: N/A (no images processed)")
    
    return embeddings_data


def save_embeddings(embeddings_data):
    """Save embeddings to a pickle file."""
    print(f"\n💾 Saving embeddings to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(embeddings_data, f)
    
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"✅ Saved {len(embeddings_data)} embeddings ({file_size_mb:.2f} MB)")


def main():
    """Main execution function."""
    print("=" * 60)
    print("  CELEBRITY FACE EMBEDDING GENERATOR")
    print("  Using face_recognition library (128-dim vectors)")
    print("=" * 60)
    
    # Check if embeddings already exist
    if os.path.exists(OUTPUT_FILE):
        response = input(f"\n⚠️  Embeddings file '{OUTPUT_FILE}' already exists. Regenerate? (y/n): ")
        if response.lower() != 'y':
            print("✅ Using existing embeddings.")
            return
    
    # Process dataset
    embeddings_data = process_dataset()
    
    if not embeddings_data:
        print("❌ No embeddings generated. Exiting.")
        return
    
    # Save embeddings
    save_embeddings(embeddings_data)
    
    print("\n" + "=" * 60)
    print("✅ Embeddings ready! You can now run: python 3_load_to_redis.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
