# 🎭 Celebrity Face Finder - Redis Vector Search Demo

An interactive demo showcasing Redis as a vector database for face similarity search. Perfect for non-technical audiences to see the "magic" of vector search!

![Redis Vector Search](https://img.shields.io/badge/Redis-Vector%20Search-DC382D?logo=redis&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive-FF4B4B?logo=streamlit&logoColor=white)

## 🎯 What This Demo Does

Upload a celebrity photo → Find similar faces in the database → See results in seconds!

**Key Features:**
- 🔍 **Vector Similarity Search** using Redis
- 🎨 **Visual Embedding Display** (128-dimensional vectors)
- ⚡ **Real-time Search** with millisecond response times
- 📊 **Interactive UI** built with Streamlit
- 🎭 **105 Celebrities** with ~17,500 images
- 💾 **Persistent Storage** (data survives restarts)

## 📁 Project Structure

```
celebrity_demo/
├── docker-compose.yml           # Redis Stack with persistence
├── requirements.txt             # Python dependencies
├── 1_download_dataset.py        # Downloads Pins celebrity dataset
├── 2_generate_embeddings.py    # Generates face embeddings (free, local)
├── 3_load_to_redis.py          # Loads embeddings into Redis
├── streamlit_app.py            # Interactive Streamlit demo
├── test_images/                # Sample celebrity photos for testing
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

- **Docker** and **Docker Compose** installed
- **Python 3.8+** installed
- **Kaggle API credentials** (for dataset download)

### Setup Kaggle API (One-time)

1. Go to https://www.kaggle.com/settings/account
2. Scroll to "API" section and click "Create New Token"
3. This downloads `kaggle.json`
4. Move it to the right location:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

### Installation & Setup (Run Once Before Lecture)

```bash
# 1. Start Redis Stack
docker-compose up -d

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download celebrity dataset (~105 celebrities, ~17,500 images)
python 1_download_dataset.py

# 4. Generate face embeddings (this may take 10-30 minutes)
python 2_generate_embeddings.py

# 5. Load embeddings into Redis
python 3_load_to_redis.py
```

**Note:** Steps 3-5 only need to be run once! The data persists in Docker volumes.

### Demo Day (Instant Start!)

```bash
# Start Redis (data already loaded)
docker-compose up -d

# Run the demo
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501` 🎉

## 🎬 Demo Workflow

### For Your Lecture

1. **Show the Streamlit UI** - Clean, professional interface
2. **Upload a test image** - Use images from `test_images/` folder
3. **Show the embedding visualization** - 128-dimensional vector as a bar chart
4. **Highlight the Redis query** - Show the actual FT.SEARCH KNN syntax
5. **Display results** - Top 5 matches with similarity scores
6. **Explain the magic** - Vector similarity, not keyword search!

### Teaching Moments

- **"This is a 128-dimensional vector"** - Show the embedding visualization
- **"Here's the actual Redis query"** - Display the FT.SEARCH syntax
- **"Search completed in X milliseconds"** - Emphasize speed
- **"94.2% match!"** - Show similarity as percentage (more intuitive)

## 📊 Dataset Information

**Pins Face Recognition Dataset**
- Source: https://www.kaggle.com/datasets/hereisburak/pins-face-recognition
- Celebrities: 105
- Total Images: ~17,500
- Includes: Tom Holland, Zendaya, Elon Musk, Taylor Swift, RDJ, Chris Evans, etc.

## 🔧 Technical Details

### Embedding Generation
- **Library:** `face_recognition` (FREE, runs locally, no API costs)
- **Model:** dlib's ResNet-based face recognition
- **Dimensions:** 128
- **Output:** Numpy array saved as pickle file

### Redis Configuration
- **Index Name:** `celebrities`
- **Algorithm:** HNSW (Hierarchical Navigable Small World)
- **Distance Metric:** COSINE
- **Fields:**
  - `name` (TAG) - Celebrity name
  - `image_path` (TEXT) - Path to image file
  - `embedding` (VECTOR) - 128-dim face vector

### Streamlit App Features
1. **Upload Section** - Upload or select test images
2. **Embedding Visualization** - Interactive Plotly chart
3. **Redis Query Display** - Show actual FT.SEARCH command
4. **Results Section** - Top K matches with photos and scores
5. **Sidebar** - Database stats and quick start guide

## 🎨 Customization

### Change Number of Results
In the Streamlit sidebar, adjust the slider (1-10 results)

### Add More Test Images
Place celebrity photos in `test_images/` folder. See `test_images/README.md` for details.

### Modify Search Parameters
Edit `streamlit_app.py`:
```python
REDIS_HOST = "localhost"  # Change if Redis is remote
REDIS_PORT = 6379         # Change if using different port
INDEX_NAME = "celebrities" # Change index name
```

## 🐛 Troubleshooting

### "Could not connect to Redis"
```bash
# Check if Redis is running
docker ps

# Start Redis
docker-compose up -d

# Check logs
docker-compose logs
```

### "No face detected"
- Use clear, front-facing photos
- Ensure good lighting
- Avoid group photos (single person works best)

### "Index not found"
```bash
# Reload data into Redis
python 3_load_to_redis.py
```

### Slow embedding generation
- This is normal! Processing 17,500 images takes time
- The embeddings are saved to `celebrity_embeddings.pkl`
- You only need to run this once

## 📦 Docker Services

### Redis Stack
- **Port 6379:** Redis server
- **Port 8001:** RedisInsight UI (visual database browser)
- **Volume:** `redis-data` (persists data between restarts)

Access RedisInsight at `http://localhost:8001` to explore the database visually!

## 🎓 Learning Resources

- [Redis Vector Search Docs](https://redis.io/docs/stack/search/reference/vectors/)
- [RediSearch Documentation](https://redis.io/docs/stack/search/)
- [Face Recognition Library](https://github.com/ageitgey/face_recognition)

## 📝 License

This is a demo project for educational purposes.

## 🙏 Credits

- **Dataset:** Pins Face Recognition Dataset by hereisburak
- **Face Recognition:** dlib and face_recognition library
- **Vector Database:** Redis Stack
- **UI Framework:** Streamlit

---

**Ready to wow your audience with Redis vector search!** 🚀✨

