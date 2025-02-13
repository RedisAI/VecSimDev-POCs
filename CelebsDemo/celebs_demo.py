import os
# Set TensorFlow logging level to ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Optionally, disable oneDNN optimizations if not needed
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from redis import Redis
from redisvl.query import VectorQuery
from redisvl.index import SearchIndex

import requests
from deepface import DeepFace
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
import base64
from io import BytesIO
from zipfile import ZipFile

# Global variables
DATASET_URL = "https://redisvl-faces-dataset.s3.us-east-1.amazonaws.com/kaggle_famous_people_dataset.zip"
DATASET_PATH = "kaggle_famous_people_dataset"

# Download and extract dataset
if not os.path.exists(DATASET_PATH):
    print("Downloading dataset...")
    response = requests.get(DATASET_URL)
    with open("dataset.zip", "wb") as f:
        f.write(response.content)
    print("Extracting dataset...")
    with ZipFile("dataset.zip", "r") as zip_ref:
        zip_ref.extractall(".")
    os.remove("dataset.zip")
    print("Dataset ready.")

def load_remote_image(url: str):
    """Download and return an image from a URL."""
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

def generate_embedding(image_path: str):
    """Generate an embedding for the image."""
    try:
        embedding = DeepFace.represent(image_path, model_name="Facenet")
        return np.array(embedding[0]["embedding"], dtype=np.float32)
    except Exception as e:
        print(f"Error generating embedding for {image_path}: {e}")
        return None

def display_images_side_by_side(images, titles, figsize=(8, 4)):
    """Display a list of images side by side."""
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    for ax, img, title in zip(axes, images, titles):
        img = img.convert("RGB")  # Convert images to RGB
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{titles[1]}.png")

celebs_embbeding = generate_embedding("kaggle_famous_people_dataset/chris_hemsworth/face_detected_891637de.jpg")
# print(celebs_embbeding)
print(f"dim = {len(celebs_embbeding)}")

SAFE_THRESHOLD=0.46
def create_redis_index(client, alg="flat"):
    """
    Define and create the Redis index using RedisVL.

    This function defines the schema for the facial recognition system,
    specifying the index name, data fields, and vector field properties.
    It uses RedisVL's `SearchIndex` to create the index with support for
    efficient vector queries. This is the cornerstone of the demo, enabling
    Redis to act as a vector database.
    """
    schema = {
        "index": {
            "name": "face_recognition",
            "prefix": "face_docs",
        },
        "fields": [
            {"name": "name", "type": "tag"},
            {"name": "photo_reference", "type": "text"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "dims": 128,
                    "distance_metric": "cosine",
                    "algorithm": alg,
                    "datatype": "float32",
                }
            }
        ]
    }
    index = SearchIndex.from_dict(schema)
    index.set_client(client)
    index.create(overwrite=False)
    return index

def inject_local_data_into_redis(base_path, index):
    """
    Load images from a local dataset, generate embeddings, and inject them into Redis.

    This function iterates through a local folder structure where each folder
    represents a unique identity (e.g., a person). For each folder, it reads an
    image, generates a vector embedding using DeepFace, and stores the data in
    Redis with the corresponding vector representation. This prepares the data
    for real-time vector search queries.
    """
    for folder_name in os.listdir(base_path)[:5]:
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(folder_path):
            continue  # Skip files, process only directories

        jpeg_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg") or f.endswith(".jpeg")]
        if not jpeg_files:
            print(f"No JPEGs found in folder: {folder_path}")
            continue

        for jpeg_file in jpeg_files:
            image_path = os.path.join(folder_path, jpeg_file)
            try:
                # Load image and convert to Base64
                with open(image_path, "rb") as img_file:
                    encoded_binary = base64.b64encode(img_file.read()).decode("utf-8")

                # Generate embedding
                embedding = generate_embedding(image_path)
                if embedding is None:
                    continue

                # Store data in Redis
                index.load([{
                    "name": folder_name,
                    "photo_reference": image_path,
                    "photo_binary": encoded_binary,
                    "embedding": embedding.tobytes()
                }])
                print(f"Stored {folder_name} in Redis with image: {jpeg_file}")
                break  # Successfully processed this folder
            except (UnidentifiedImageError, IOError) as e:
                print(f"Error processing image {image_path}: {e}")
                continue

def query_redis(target_image_path, index, client, threshold=SAFE_THRESHOLD):
    """
    Perform a vector similarity search in Redis and display visual results.

    This function takes a target image, generates its vector embedding,
    and queries Redis using RedisVL's `VectorQuery`. The query retrieves
    the closest match from the index, calculates the similarity score
    (distance), and compares it against a threshold. It then displays the
    target image alongside the closest match or indicates if no match is found.
    """
    # Generate embedding for the target image
    target_embedding = generate_embedding(target_image_path)
    if target_embedding is None:
        print(f"Failed to generate embedding for {target_image_path}")
        return

    # Query Redis
    query = VectorQuery(
        vector=target_embedding.tolist(),
        vector_field_name="embedding",
        return_fields=["name", "photo_reference", "vector_distance", "photo_binary"],
        num_results=1  # Only need the best match
    )
    results = index.query(query)

    if not results:
        print("No matches found in Redis.")
        return

    # Parse the best match
    best_match = results[0]
    match_name = best_match["name"]
    match_distance = float(best_match["vector_distance"])
    match_image = Image.open(BytesIO(base64.b64decode(best_match["photo_binary"]))).convert("RGB")

    # Load the target image and ensure RGB mode
    target_image = load_remote_image(target_image_path).convert("RGB")

    # Display results
    if match_distance > threshold:
        print(f"\nNo match found. Closest match is {match_name} (Distance: {match_distance:.2f}).")
        display_images_side_by_side(
            [target_image, match_image],
            ["Target Image", f"Closest Match: {match_name} (Not Found)"]
        )
    else:
        print(f"\nMatch found: {match_name}, Distance: {match_distance:.2f}")
        display_images_side_by_side(
            [target_image, match_image],
            ["Target Image", f"Best Match: {match_name}"]
        )

client = Redis(host="localhost", port=6379)
client.flushall()
# Ensure the RedisVL index is valid
index = create_redis_index(client)

# print(index.schema)

# Check if Redis already contains data
indexed_faces_count = index.info()['num_docs']
if indexed_faces_count > 0:
    print(f"Redis already contains {indexed_faces_count} records. Skipping data injection.")
else:
    # Inject data into Redis from a local dataset if no data is present
    dataset_path = "kaggle_famous_people_dataset"
    inject_local_data_into_redis(dataset_path, index)
    print("Data successfully injected into Redis.")
    indexed_faces_count = index.info()['num_docs']
    print(f"Redis now contains {indexed_faces_count} records.")

# Test queries
test_cases = [
    ("https://people.com/thmb/cS-3Y34QFwEbRO_x50acJP3MwbQ=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(734x348:736x350)/Tom-Hanks-That-Thing-You-Do-110624-NA-tout-d517a235093747949aec98449b8b9245.jpg", "Tom Hanks"),
    ("https://github.com/serengil/deepface/raw/master/tests/dataset/img2.jpg", "Angelina Jolie"),
    ("https://m.media-amazon.com/images/M/MV5BOGY5NTNiMmUtMjdiYi00ZmZkLTg3OTgtNDQ1OTVlZWUzY2IzXkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg", "Seth Rogan"),
    ("https://media.hugogloss.uol.com.br/uploads/2023/10/Kristen-Stewart-617x347.png", "Kristen Stewart"),
    ("https://people.com/thmb/6lv2ts3_inac7CLVSoMYSAUGwow=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(749x0:751x2)/cher-fw-tout-1010-fa15ee6f98824650a2f92f6e9665b7af.jpg", "cher"),
    ("https://static.wikia.nocookie.net/littlewomen/images/a/ac/Emmawatson.png/revision/latest?cb=20191221175400", "Emma Watson"),
]

# Run facial recognition
for image_url, label in test_cases:
    print(f"\n--- Testing: {label} ---")
    query_redis(image_url, index, client, threshold=SAFE_THRESHOLD)
