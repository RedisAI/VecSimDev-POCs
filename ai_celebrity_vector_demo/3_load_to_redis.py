#!/usr/bin/env python3
"""
Load celebrity embeddings into Redis with vector search index.

This script:
1. Loads embeddings from the pickle file
2. Creates a Redis index with HNSW algorithm
3. Stores celebrity data (name, image_path, embedding) in Redis

Redis Schema:
- Index name: celebrities
- Fields: name (tag), image_path (text), embedding (vector)
- Vector: 128 dimensions, COSINE distance, HNSW algorithm
"""

import os
import pickle
import redis
from redis.commands.search.field import TextField, TagField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from tqdm import tqdm
import numpy as np

# Configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
EMBEDDINGS_FILE = "celebrity_embeddings.pkl"
INDEX_NAME = "celebrities"
KEY_PREFIX = "celeb:"
VECTOR_DIM = 128


def connect_redis():
    """Connect to Redis server."""
    try:
        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=False  # We need binary for vectors
        )
        client.ping()
        print(f"✅ Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        return client
    except redis.ConnectionError:
        print(f"❌ Could not connect to Redis at {REDIS_HOST}:{REDIS_PORT}")
        print("   Make sure Redis is running: docker-compose up -d")
        return None


def create_index(client):
    """
    Create Redis search index for celebrity face vectors.
    
    Uses HNSW (Hierarchical Navigable Small World) algorithm for
    efficient approximate nearest neighbor search.
    """
    try:
        # Drop existing index if it exists
        try:
            client.ft(INDEX_NAME).dropindex(delete_documents=True)
            print(f"🗑️  Dropped existing index: {INDEX_NAME}")
        except:
            pass
        
        # Define schema
        schema = (
            TagField("name"),                    # Celebrity name (searchable tag)
            TextField("image_path"),             # Path to image file
            VectorField("embedding",             # 128-dim face embedding
                "HNSW", {
                    "TYPE": "FLOAT32",
                    "DIM": VECTOR_DIM,
                    "DISTANCE_METRIC": "COSINE",
                }
            ),
        )
        
        # Create index
        client.ft(INDEX_NAME).create_index(
            fields=schema,
            definition=IndexDefinition(
                prefix=[KEY_PREFIX],
                index_type=IndexType.HASH
            )
        )
        
        print(f"✅ Created index: {INDEX_NAME}")
        print(f"   Algorithm: HNSW")
        print(f"   Distance Metric: COSINE")
        print(f"   Vector Dimensions: {VECTOR_DIM}")
        
    except Exception as e:
        print(f"❌ Error creating index: {e}")
        raise


def load_embeddings():
    """Load embeddings from pickle file."""
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"❌ Embeddings file not found: {EMBEDDINGS_FILE}")
        print("   Please run: python 2_generate_embeddings.py")
        return None
    
    print(f"📂 Loading embeddings from {EMBEDDINGS_FILE}...")
    
    with open(EMBEDDINGS_FILE, 'rb') as f:
        embeddings_data = pickle.load(f)
    
    print(f"✅ Loaded {len(embeddings_data)} embeddings")
    return embeddings_data


def store_in_redis(client, embeddings_data):
    """Store celebrity embeddings in Redis."""
    print(f"\n💾 Storing {len(embeddings_data)} embeddings in Redis...")
    
    pipeline = client.pipeline(transaction=False)
    batch_size = 100
    
    for idx, data in enumerate(tqdm(embeddings_data, desc="Storing")):
        key = f"{KEY_PREFIX}{idx}"
        
        # Prepare data
        embedding_bytes = np.array(data['embedding'], dtype=np.float32).tobytes()
        
        # Store as Redis hash
        pipeline.hset(
            key,
            mapping={
                "name": data['name'],
                "image_path": data['image_path'],
                "embedding": embedding_bytes
            }
        )
        
        # Execute batch
        if (idx + 1) % batch_size == 0:
            pipeline.execute()
    
    # Execute remaining
    pipeline.execute()
    
    print(f"✅ Stored {len(embeddings_data)} celebrity embeddings")


def verify_index(client):
    """Verify the index was created successfully."""
    try:
        info = client.ft(INDEX_NAME).info()
        num_docs = info['num_docs']
        print(f"\n📊 Index Statistics:")
        print(f"   Index Name: {INDEX_NAME}")
        print(f"   Documents: {num_docs}")
        print(f"   Status: Ready for queries! 🚀")
    except Exception as e:
        print(f"⚠️  Could not verify index: {e}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("  REDIS VECTOR DATABASE LOADER")
    print("=" * 60)
    
    # Connect to Redis
    client = connect_redis()
    if not client:
        return
    
    # Load embeddings
    embeddings_data = load_embeddings()
    if not embeddings_data:
        return
    
    # Create index
    print("\n🔧 Creating Redis search index...")
    create_index(client)
    
    # Store embeddings
    store_in_redis(client, embeddings_data)
    
    # Verify
    verify_index(client)
    
    print("\n" + "=" * 60)
    print("✅ Redis database ready! You can now run: streamlit run app.py")
    print("=" * 60)
    print("\n💡 Tip: Data is persisted in Docker volume.")
    print("   Next time just run: docker-compose up -d && streamlit run app.py")


if __name__ == "__main__":
    main()
