# ⚡ Quick Start - Celebrity Face Finder

## 🎯 Goal
Get the demo running in 2 commands on demo day!

---

## 📋 Prerequisites (One-Time Setup)

### 1. Install Docker
- Download from https://www.docker.com/
- Make sure it's running

### 2. Install Python 3.8+
```bash
python3 --version  # Should be 3.8 or higher
```

### 3. Setup Kaggle API
```bash
# Get your API key from https://www.kaggle.com/settings/account
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## 🚀 First Time Setup (30-60 minutes)

### Option A: Automated (Recommended)
```bash
./setup.sh
```

### Option B: Manual
```bash
# 1. Start Redis
docker-compose up -d

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset
python 1_download_dataset.py

# 4. Generate embeddings (takes 10-30 min)
python 2_generate_embeddings.py

# 5. Load to Redis
python 3_load_to_redis.py
```

---

## 🎬 Demo Day (2 Commands!)

```bash
docker-compose up -d
streamlit run streamlit_app.py
```

**Opens at:** http://localhost:8501

---

## 🎭 During the Demo

1. **Upload** a celebrity photo (or select from test_images)
2. **Show** the embedding visualization
3. **Point out** the Redis query
4. **Display** the results with similarity scores
5. **Emphasize** the speed!

---

## 🛑 Stop Everything

```bash
docker-compose down
```

---

## 🐛 Troubleshooting

### Redis won't connect?
```bash
docker-compose down
docker-compose up -d
# Wait 5 seconds, try again
```

### No face detected?
- Use a clear, front-facing photo
- Single person (not a group)
- Good lighting

### Slow first search?
- Normal! Model is loading
- Subsequent searches are fast

---

## 📚 More Info

- **Full docs:** `celebrity_demo_README.md`
- **Demo guide:** `DEMO_GUIDE.md`
- **Project summary:** `PROJECT_SUMMARY.md`

---

## ✅ Checklist

Before your lecture:
- [ ] Run `./setup.sh` successfully
- [ ] Test the Streamlit app
- [ ] Add 3-5 test images to `test_images/`
- [ ] Practice the demo flow
- [ ] Read `DEMO_GUIDE.md`

Demo day:
- [ ] Start Docker
- [ ] Run `docker-compose up -d`
- [ ] Run `streamlit run streamlit_app.py`
- [ ] Open http://localhost:8501
- [ ] Wow your audience! 🎉

---

**That's it! You're ready to demo Redis vector search!** 🚀

