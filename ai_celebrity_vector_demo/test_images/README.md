# Test Images

This folder contains test celebrity photos for the demo.

## Purpose

These are **different photos** of celebrities that are IN the database but NOT part of the training images. This lets you demonstrate:

> "Here's a different photo of Tom Holland - watch it find him!"

## How to Add Test Images

1. **Find celebrities in your dataset**:
   ```bash
   ls pins_dataset/
   ```

2. **Download different photos** of those celebrities from Google Images

3. **Save them here** with descriptive names:
   - `tom_holland_test.jpg`
   - `zendaya_test.jpg`
   - `elon_musk_test.jpg`
   - `taylor_swift_test.jpg`
   - `rdj_test.jpg`
   - etc.

## Recommended Test Images

For the Pins dataset, good celebrities to test:
- Tom Holland
- Zendaya
- Elon Musk
- Taylor Swift
- Robert Downey Jr.
- Chris Evans
- Scarlett Johansson
- Emma Watson
- Leonardo DiCaprio
- Jennifer Lawrence

## Tips

- Use **clear, front-facing photos** for best results
- Avoid group photos (single person works best)
- JPG or PNG format
- Reasonable size (not too large, 500KB-2MB is fine)

## Quick Download Script

You can use this Python script to download test images:

```python
import requests
from pathlib import Path

# Example: Download a test image
def download_image(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        Path(filename).write_bytes(response.content)
        print(f"✅ Downloaded {filename}")

# Add your own URLs here
test_images = [
    ("https://example.com/tom_holland.jpg", "tom_holland_test.jpg"),
    # Add more...
]

for url, filename in test_images:
    download_image(url, filename)
```

## During the Demo

The Streamlit app will automatically detect images in this folder and show them in a dropdown selector. This makes it easy to quickly demo different celebrities without uploading files!

