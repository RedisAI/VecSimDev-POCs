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
from zipfile import ZipFile
import json
# Global variables
DATASET_URL = "https://redisvl-faces-dataset.s3.us-east-1.amazonaws.com/kaggle_famous_people_dataset.zip"
DATASET_PATH = "Celeb1000"
MAX_DOCS = 30000
SAFE_THRESHOLD=0.99
GLOBAL=True

# Download and extract dataset
def download_faces_dataset():
    """Download and extract the dataset if not already present."""
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
                continue

            # Store data in Redis
            index.load([{
                "name": celeb_name,
                "photo_reference": image_path,
                "photo_binary": encoded_binary,
                "embedding": embedding.tobytes()
            }])
            if num_docs % 100 == 0:
                print(f"Stored {num_docs} docs in Redis")
            num_docs += 1
            if num_docs > max_docs:
                break
        except (UnidentifiedImageError, IOError) as e:
            print(f"Error processing image url for {celeb_name}: {e}")


def inject_local_data_into_redis(base_path, index, max_docs=MAX_DOCS, skip_if_name_exists=False, force_skip=False):
    """
    Load images from a local dataset, generate embeddings, and inject them into Redis.

    This function iterates through a local folder structure where each folder
    represents a unique identity (e.g., a person). For each folder, it reads an
    image, generates a vector embedding using DeepFace, and stores the data in
    Redis with the corresponding vector representation. This prepares the data
    for real-time vector search queries.
    """
    dataset_metadata = None
    celebs_metadata = None
    max_pics_per_celeb = 1000
    if os.path.exists(os.path.join(base_path, "metadata.json")):
        dataset_metadata = json.load(open(os.path.join(base_path, "metadata.json")))
        max_pics_per_celeb = dataset_metadata["statistics"]["max_num_pics"]
        celebs_metadata = dataset_metadata["celebs"]
    done = False
    num_docs = 0
    num_celebs = 0
    for folder_name in os.listdir(base_path):
        if done:
            break
        num_celebs += 1
        if num_celebs % 100 == 0:
            print(f"Processed {num_celebs} celebs and total of {num_docs} docs")
        if skip_if_name_exists:
            # search for the name in the dataset
            try:
                res = client.ft("face_recognition").search(Query(folder_name).no_content().paging(0, max_pics_per_celeb))
            except redis.exceptions.ResponseError as e:
                print(f"Error searching for {folder_name} failed: {e}")
                continue
            # if the number of results equals the expected number of pics according to the metadata
            # file, skip
            if res.total > 0:
                if force_skip:
                    print(f"force_skip is on. {folder_name} has {res.total} docs. just continue to the next celeb...")
                    continue
                if celebs_metadata is not None:
                    if res.total == celebs_metadata[folder_name]["num_pics"]:
                        continue
                    else:
                        print(f"Found {res.total} pics for {folder_name} in the dataset, but metadata shows {celebs_metadata[folder_name]['num_pics']}. Will remove current and re-add to Redis.")
                else:
                    print("No metadata found, can't skip. ")
                # we need to query again to get all the hashes
                if (res.total < len(res.docs)):
                    res = client.ft("face_recognition").search(Query(folder_name).no_content().paging(0, res.total))
                print(f"Removing {res.total} docs for {folder_name} from Redis")
                for key in res.docs:
                    client.delete(key.id)
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

                num_docs += 1
                if num_docs > max_docs:
                    done = True
                # break  # Successfully processed this folder
            except (UnidentifiedImageError, IOError) as e:
                print(f"Error processing image {image_path}: {e}")
                continue

def query_redis(target_image_path, index, label, threshold=SAFE_THRESHOLD, num_results=1):
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
            [f"{label}", f"Best Match: {match_name}, Similarity: {1 - match_distance:.2f}"]
        )

client = Redis(host="localhost", port=6379)
    # Ensure the RedisVL index is valid
index = create_redis_index(client, alg="hnsw")
def initialize_all(reset = False, skip_if_name_exists=True, force_skip=False):
    if reset:
        client.flushall()
    initial_db_size = client.dbsize()
    print(f"current db size: {initial_db_size}")
    index = create_redis_index(client, alg="hnsw")

    # Check if Redis already contains data
    # indexed_faces_count = index.info()['num_docs']
    # if indexed_faces_count > 0:
    #     print(f"Redis already contains {indexed_faces_count} records. Skipping data injection.")
    # Inject data into Redis from a local dataset if no data is present
    dataset_path = DATASET_PATH
    inject_local_data_into_redis(dataset_path, index, skip_if_name_exists=skip_if_name_exists, force_skip=force_skip)
    curr_db_size = client.dbsize()
    new_docs = curr_db_size - initial_db_size
    print(f"successfully injected {new_docs} docs into Redis.")
    # indexed_faces_count = index.info()['num_docs']
    print(f"Redis now contains {curr_db_size} records.")


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
    ("/workspaces/Code/VecSimDev-POCs/CelebsDemo/uploads/Mwe.jpg", "Mwe"),
    ("https://people.com/thmb/6lv2ts3_inac7CLVSoMYSAUGwow=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(749x0:751x2)/cher-fw-tout-1010-fa15ee6f98824650a2f92f6e9665b7af.jpg", "cher"),
    ("https://people.com/thmb/cS-3Y34QFwEbRO_x50acJP3MwbQ=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(734x348:736x350)/Tom-Hanks-That-Thing-You-Do-110624-NA-tout-d517a235093747949aec98449b8b9245.jpg", "Tom Hanks"),
    ("https://github.com/serengil/deepface/raw/master/tests/dataset/img2.jpg", "Angelina Jolie"),
    ("https://m.media-amazon.com/images/M/MV5BOGY5NTNiMmUtMjdiYi00ZmZkLTg3OTgtNDQ1OTVlZWUzY2IzXkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg", "Seth Rogan"),
    ("https://media.hugogloss.uol.com.br/uploads/2023/10/Kristen-Stewart-617x347.png", "Kristen Stewart"),
    ("https://static.wikia.nocookie.net/littlewomen/images/a/ac/Emmawatson.png/revision/latest?cb=20191221175400", "Emma Watson"),
]
# Run facial recognition
# initialize_all(reset=False, skip_if_name_exists=True, force_skip=True)
for image_url, label in test_cases[:1]:
    print(f"\n--- Testing: {label} ---")
    query_redis(image_url, index, label, threshold=SAFE_THRESHOLD, num_results=1)
