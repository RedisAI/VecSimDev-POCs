import os
# Set TensorFlow logging level to ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Optionally, disable oneDNN optimizations if not needed
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import redis
from redis import Redis
from redisvl.query import VectorQuery
from redisvl.index import SearchIndex
from redis.commands.search.query import Query

import requests
from deepface import DeepFace
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError, ImageOps
import base64
from io import BytesIO
import json
# Global variables
DATASET_URL = "https://redisvl-faces-dataset.s3.us-east-1.amazonaws.com/kaggle_famous_people_dataset.zip"
DATASET_PATH = "Celeb1000"
MAX_DOCS = 30000
SAFE_THRESHOLD=0.99

def load_remote_image(url: str):
    """Download and return an image from a URL."""

    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

def generate_embedding(image_path: str):
    """Generate an embedding for the image."""
    try:
        embedding = DeepFace.represent(image_path, model_name="Facenet", enforce_detection=False)
        return np.array(embedding[0]["embedding"], dtype=np.float32)
    except Exception as e:
        print(f"Error generating embedding for {image_path}: {e}")
        return None

def display_images_side_by_side(images, titles, figsize=(8, 4), show=True):
    """Display a list of images side by side."""
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    for ax, img, title in zip(axes, images, titles):
        img = img.convert("RGB")  # Convert images to RGB
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=12)
    if (show):
        plt.tight_layout()
        plt.savefig(f"res.png")
        os.system(f"code res.png")

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

def query_redis(target_image_path, index, client, threshold=SAFE_THRESHOLD, num_results=1):
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
        num_results=num_results  # Only need the best match
    )
    results = index.query(query)

    if not results:
        print("No matches found in Redis.")
        return

    if (num_results > 1):
        print("Closest matches:")
        for result in results:
            print(f"Distance: {float(result['vector_distance']):.2f}, Name: {result['name']}")

        return
    # Parse the best match
    best_match = results[0]
    match_name = best_match["name"]
    match_distance = float(best_match["vector_distance"])
    match_image = Image.open(BytesIO(base64.b64decode(best_match["photo_binary"]))).convert("RGB")

    # # Load the target image and ensure RGB mode
    # target_image = load_remote_image(target_image_path).convert("RGB")
    # Check if target_image_path is a URL or a local file path
    if target_image_path.startswith("http://") or target_image_path.startswith("https://"):
        # Load image from URL
        target_image = load_remote_image(target_image_path)
    else:
        # Load image from local file path
        try:
            target_image = Image.open(target_image_path)
            target_image = ImageOps.exif_transpose(target_image)
        except (UnidentifiedImageError, IOError) as e:
            print(f"Error loading image from {target_image_path}: {e}")
            return
    # Display results
    if match_distance > threshold:
        print(f"\nNo match found. Closest match is {match_name} (Distance: {match_distance:.2f}).")
        display_images_side_by_side(
            [target_image, match_image],
            ["Target Image", f"Closest Match: {match_name} (Not Found)"],
            show=False
        )
    else:
        print(f"\nMatch found: {match_name}, Similarity: {1 - match_distance:.2f}")
        display_images_side_by_side(
            [target_image, match_image],
            ["Target_Image", f"Best_Match: {match_name}"]
        )

def index_url_images(image_urls, max_docs=MAX_DOCS):
    num_docs = 0
    for url, celeb_name in image_urls:
        try:
            # Load image from URL
            image = load_remote_image(url)
            # Convert image to Base64
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            encoded_binary = base64.b64encode(buffered.getvalue()).decode("utf-8")

            image_path = f"{celeb_name}.jpg"
            image.save(image_path)
            # Generate embedding
            embedding = generate_embedding(image_path)
            if embedding is None:
                print(f"Failed to generate embedding for {celeb_name}")
                return False
                continue

            # Store data in Redis
            index.load([{
                "name": celeb_name,
                "photo_reference": image_path,
                "photo_binary": encoded_binary,
                "embedding": embedding.tobytes()
            }])
            print(f"Stored {celeb_name} in Redis.")
            num_docs += 1
            if num_docs > max_docs:
                break
        except (UnidentifiedImageError, IOError) as e:
            print(f"Error processing image url for {celeb_name}: {e}")

# ANSI escape codes for formatting
BOLD = "\033[1m"
BLUE = "\033[94m"
GREEN = "\033[92m"
CYAN = "\033[96m"
RESET = "\033[0m"
client = Redis(host="localhost", port=6090)
    # Ensure the RedisVL index is valid
client.flushall()
input(f"\n{BOLD}{CYAN}✨ Welcome to the Redis Vector Similarity Search Demo! ✨{RESET}\nPress Enter to get started...")
print(f"\n{BOLD}{BLUE}🔄 Initializing the Redis index...{RESET}")
index = create_redis_index(client, alg="hnsw")
print(f"{GREEN}✅ Index created successfully!{RESET} Now, let's insert some data.\n")


cont = "y"
while  cont.lower() == "y":
    print(f"\n{BOLD}{CYAN}📝 Adding a new celebrity image...{RESET}")
    celeb_name = input(f"{BOLD}👤 Enter the celebrity's name:\n> {RESET}")
    img_url = input(f"{BOLD}🖼️ Provide an image URL:\n> {RESET}")
    index_url_images([(img_url, celeb_name)])
    cont = input(f"\n➕ {BOLD}Do you want to add another image? (y/n){RESET}\n> ")

curr_db_size = client.dbsize()
print(f"\n{BOLD}{GREEN}🎉 Successfully inserted {curr_db_size} images into Redis!{RESET}\n")

add_more = [
    ("https://www.news08.net/wp-content/uploads/2019/01/%D7%A9%D7%9C%D7%95%D7%9E%D7%99%D7%AA-%D7%9E%D7%9C%D7%9B%D7%94.jpg", "Shlomit Malka"),
    ("https://media3.reshet.tv/image/upload/t_grid-item-large/v1681985190/uploads/2023/903500455.webp", "Noa Kirel"),
    ("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRo35PgMDejsYoGgOIp11Mf2KsMtfABKjv1iw&s", "Anna Zak"),
    ("https://mediaslide-europe.storage.googleapis.com/uno/pictures/4214/98417/profile-1706876576-31f53c8942be116c4435e3cd0894c02b.jpg?v=1706876639", "Anna Zak"),
    ("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR6biVTSaMARHDs5QC-8rI4l3h-byP6rOpptw&s", "Gal Gadot"),
    ("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS9__mOq47LRDKb9xiLMQ9JpUvHsieGt6xYQg&s", "Ran Danker"),
]
# index_url_images(add_more)

# Test queries
test_cases = [
    ("https://media.reshet.tv/image/upload/t_grid-item-large/v1692687056/uploads/2023/903679549.webp", "Me"),
    ("https://people.com/thmb/6lv2ts3_inac7CLVSoMYSAUGwow=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(749x0:751x2)/cher-fw-tout-1010-fa15ee6f98824650a2f92f6e9665b7af.jpg", "cher"),
    ("https://people.com/thmb/cS-3Y34QFwEbRO_x50acJP3MwbQ=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(734x348:736x350)/Tom-Hanks-That-Thing-You-Do-110624-NA-tout-d517a235093747949aec98449b8b9245.jpg", "Tom Hanks"),
    ("https://github.com/serengil/deepface/raw/master/tests/dataset/img2.jpg", "Angelina Jolie"),
    ("https://m.media-amazon.com/images/M/MV5BOGY5NTNiMmUtMjdiYi00ZmZkLTg3OTgtNDQ1OTVlZWUzY2IzXkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg", "Seth Rogan"),
    ("https://media.hugogloss.uol.com.br/uploads/2023/10/Kristen-Stewart-617x347.png", "Kristen Stewart"),
    ("https://static.wikia.nocookie.net/littlewomen/images/a/ac/Emmawatson.png/revision/latest?cb=20191221175400", "Emma Watson"),
]

print(f"\n{BOLD}{BLUE}📊 Database is ready! Time for some queries!{RESET}")

celeb_name = input(f"{BOLD}👤 Enter the celebrity's name:\n> {RESET}")
img_url = input(f"{BOLD}🖼️ Provide an image URL:\n> {RESET}")

print(f"\n{BOLD}{CYAN}🔍 Searching for similar images to: {celeb_name}{RESET}")
query_redis(img_url, index, client, threshold=SAFE_THRESHOLD, num_results=1)
