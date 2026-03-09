# 🎭 AI Celebrity Vector Demo

An interactive demo showcasing **Redis as a vector database** for face similarity search using AI embeddings.

Perfect for demonstrating the "magic" of vector search to non-technical audiences!

---

## ⚡ Quick Start

### First Time Setup (30-60 minutes)

```bash
./setup.sh
```

### Demo Day (Instant!)

```bash
docker-compose up -d
streamlit run streamlit_app.py
```

**Opens at:** http://localhost:8501

---

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 2 commands
- **[celebrity_demo_README.md](celebrity_demo_README.md)** - Complete documentation
- **[DEMO_GUIDE.md](DEMO_GUIDE.md)** - Demo day reference with talking points
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Overview of all files

---

## 🎯 What This Demo Does

1. **Upload** a celebrity photo
2. **Generate** a 128-dimensional face embedding (AI vector)
3. **Search** Redis for similar faces using vector similarity
4. **Display** top matches with similarity scores in milliseconds

---

## ✨ Features

- 🔍 **Vector Similarity Search** using Redis
- 🎨 **Visual Embedding Display** (128-dimensional vectors)
- ⚡ **Real-time Search** with millisecond response times
- 📊 **Interactive UI** built with Streamlit
- 🎭 **105 Celebrities** with ~17,500 images
- 💾 **Persistent Storage** (data survives restarts)
- 🎉 **Balloons animation** for high-confidence matches

---

## 📁 Project Structure

```
ai_celebrity_vector_demo/
├── 1_download_dataset.py        # Downloads Pins celebrity dataset
├── 2_generate_embeddings.py    # Generates 128-dim face vectors
├── 3_load_to_redis.py          # Loads embeddings into Redis
├── streamlit_app.py            # Interactive demo app ⭐
├── docker-compose.yml          # Redis Stack with persistence
├── requirements.txt            # Python dependencies
├── setup.sh                    # One-command setup
├── test_images/                # Sample celebrity photos
└── README.md                   # This file
```

---

## 🔧 Technical Stack

- **Database:** Redis Stack (HNSW index, COSINE distance)
- **AI Model:** face_recognition library (128-dim embeddings, FREE, local)
- **UI:** Streamlit (interactive web app)
- **Visualization:** Plotly (beautiful charts)
- **Dataset:** Pins Face Recognition (105 celebrities, 17.5K images)
- **Infrastructure:** Docker Compose (with persistence)

---

## 🎬 Demo Flow

1. Upload a celebrity photo
2. Show the 128-dimensional embedding visualization
3. Display the Redis FT.SEARCH query
4. Present top 5 matches with similarity scores
5. Emphasize the speed (milliseconds!)

---

## 🐛 Troubleshooting

### Redis won't connect?
```bash
docker-compose down
docker-compose up -d
```

### No face detected?
- Use clear, front-facing photos
- Single person (not a group)
- Good lighting

### Need to reload data?
```bash
python 3_load_to_redis.py
```

---

## 📊 Dataset

**Pins Face Recognition Dataset**
- Source: https://www.kaggle.com/datasets/hereisburak/pins-face-recognition
- Celebrities: 105
- Total Images: ~17,500
- Includes: Tom Holland, Zendaya, Elon Musk, Taylor Swift, RDJ, Chris Evans, etc.

---

## 🎓 What You'll Learn

- How vector embeddings represent faces
- How Redis performs vector similarity search
- HNSW algorithm for efficient nearest neighbor search
- COSINE distance for measuring similarity
- Real-world AI/ML application with Redis

---

**Ready to wow your audience with Redis vector search!** 🚀✨

For detailed instructions, see [celebrity_demo_README.md](celebrity_demo_README.md)

