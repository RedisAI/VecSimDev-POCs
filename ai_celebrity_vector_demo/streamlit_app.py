#!/usr/bin/env python3
"""
Celebrity Face Finder - Interactive Demo

An interactive Streamlit app demonstrating Redis vector search for face similarity.
Upload a photo and find matching celebrities in the database!
"""

import streamlit as st
import redis
import face_recognition
import numpy as np
from PIL import Image
import os
import time
from pathlib import Path
import plotly.graph_objects as go

# Configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
INDEX_NAME = "celebrities"
TEST_IMAGES_DIR = "test_images"
VECTOR_DIM = 128

# Page config
st.set_page_config(
    page_title="Celebrity Face Finder",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF4B4B;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .query-box {
        background-color: #1e1e1e;
        color: #00ff00;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        overflow-x: auto;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_redis_client():
    """Get Redis client (cached)."""
    try:
        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=False
        )
        client.ping()
        return client
    except redis.ConnectionError:
        st.error(f"❌ Could not connect to Redis at {REDIS_HOST}:{REDIS_PORT}")
        st.info("Make sure Redis is running: `docker-compose up -d`")
        return None


def generate_embedding(image):
    """Generate 128-dim face embedding from PIL Image."""
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Generate embedding
        encodings = face_recognition.face_encodings(img_array)
        
        if len(encodings) == 0:
            return None
        
        return encodings[0]
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None


def visualize_embedding(embedding):
    """Create a visual representation of the 128-dim embedding."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(range(len(embedding))),
        y=embedding,
        marker=dict(
            color=embedding,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Value")
        ),
        hovertemplate='Dimension: %{x}<br>Value: %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="128-Dimensional Face Embedding Vector",
        xaxis_title="Dimension",
        yaxis_title="Value",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def search_redis(client, embedding, top_k=5):
    """
    Search Redis for similar faces using vector similarity.
    
    Returns: list of (name, image_path, distance) tuples
    """
    try:
        # Convert embedding to bytes
        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
        
        # Build KNN query
        query = f"*=>[KNN {top_k} @embedding $vec AS distance]"
        
        # Execute search
        start_time = time.time()
        results = client.ft(INDEX_NAME).search(
            query,
            query_params={"vec": embedding_bytes}
        )
        search_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Parse results
        matches = []
        for doc in results.docs:
            name = doc.name.decode('utf-8')
            image_path = doc.image_path.decode('utf-8')
            distance = float(doc.distance)
            matches.append((name, image_path, distance))
        
        return matches, search_time
    
    except Exception as e:
        st.error(f"Search error: {e}")
        return [], 0


def display_redis_query(top_k=5):
    """Display the Redis FT.SEARCH query being executed."""
    query_text = f"""FT.SEARCH {INDEX_NAME} "*=>[KNN {top_k} @embedding $vec AS distance]"
  PARAMS 2 vec <128-dim-vector-bytes>
  RETURN 3 name image_path distance
  SORTBY distance ASC
  LIMIT 0 {top_k}"""

    st.markdown("### 🔍 Redis Query")
    st.markdown(f'<div class="query-box">{query_text}</div>', unsafe_allow_html=True)


def main():
    """Main Streamlit app."""

    # Header
    st.markdown('<div class="main-header">🎭 Celebrity Face Finder</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Powered by Redis Vector Search</div>', unsafe_allow_html=True)

    # Connect to Redis
    client = get_redis_client()
    if not client:
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)

        st.markdown("---")
        st.header("📊 Database Info")
        try:
            info = client.ft(INDEX_NAME).info()
            num_docs = info['num_docs']
            st.metric("Total Celebrities", num_docs)
            st.metric("Vector Dimensions", VECTOR_DIM)
            st.metric("Distance Metric", "COSINE")
            st.metric("Algorithm", "HNSW")
        except:
            st.warning("Index not found. Run setup scripts first.")

        st.markdown("---")
        st.markdown("### 🚀 Quick Start")
        st.code("""
# Setup (once)
docker-compose up -d
pip install -r requirements.txt
python 1_download_dataset.py
python 2_generate_embeddings.py
python 3_load_to_redis.py

# Run demo
streamlit run streamlit_app.py
        """, language="bash")

    # Main content
    st.markdown("---")

    # Upload section
    st.header("📸 Upload or Select an Image")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a celebrity photo...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a photo of a celebrity to find similar faces"
        )

    with col2:
        # Test images selector
        if os.path.exists(TEST_IMAGES_DIR):
            test_images = [f for f in os.listdir(TEST_IMAGES_DIR)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if test_images:
                selected_test = st.selectbox(
                    "Or select a test image:",
                    [""] + test_images
                )
                if selected_test:
                    uploaded_file = open(os.path.join(TEST_IMAGES_DIR, selected_test), 'rb')

    # Process image
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        st.markdown("---")

        # Display uploaded image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        # Generate embedding
        with st.spinner("🔄 Generating face embedding..."):
            embedding = generate_embedding(image)

        if embedding is None:
            st.error("❌ No face detected in the image. Please upload a clear photo with a visible face.")
            st.stop()

        st.success("✅ Face detected and embedding generated!")

        # Visualize embedding
        st.markdown("---")
        st.header("🧬 Face Embedding Visualization")
        st.plotly_chart(visualize_embedding(embedding), use_container_width=True)

        with st.expander("ℹ️ What is a face embedding?"):
            st.markdown("""
            A **face embedding** is a 128-dimensional vector that represents the unique features of a face.

            - Each dimension captures different facial characteristics (eyes, nose, jawline, etc.)
            - Similar faces have similar embedding vectors
            - We use **cosine similarity** to measure how close two faces are
            - This is the "magic" behind face recognition! 🎩✨
            """)

        # Display Redis query
        st.markdown("---")
        display_redis_query(top_k)

        # Search Redis
        st.markdown("---")
        st.header("🎯 Search Results")

        with st.spinner("🔍 Searching Redis for similar faces..."):
            matches, search_time = search_redis(client, embedding, top_k)

        if not matches:
            st.warning("No matches found.")
            st.stop()

        # Display search time
        st.success(f"⚡ Search completed in **{search_time:.2f} ms**")

        # Display results
        st.markdown(f"### Top {len(matches)} Matches")

        for idx, (name, image_path, distance) in enumerate(matches, 1):
            # Calculate similarity percentage (1 - cosine distance)
            similarity = (1 - distance) * 100

            col1, col2 = st.columns([1, 2])

            with col1:
                # Display celebrity image
                if os.path.exists(image_path):
                    celeb_img = Image.open(image_path)
                    st.image(celeb_img, use_container_width=True)
                else:
                    st.warning("Image not found")

            with col2:
                st.markdown(f"### #{idx} {name}")
                st.metric("Similarity Score", f"{similarity:.2f}%")
                st.metric("Distance", f"{distance:.4f}")

                # Add confetti for high matches
                if similarity >= 90 and idx == 1:
                    st.balloons()

                st.progress(similarity / 100)

            st.markdown("---")


if __name__ == "__main__":
    main()
