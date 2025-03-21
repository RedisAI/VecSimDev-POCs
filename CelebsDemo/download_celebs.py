import os
# Set TensorFlow logging level to ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Optionally, disable oneDNN optimizations if not needed
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from datasets import load_dataset
from PIL import Image
import json
from unidecode import unidecode


MAX_CELEBS = 1000
celebs_dir = "Celeb1000"
metadata_dir = os.path.join(celebs_dir, "metadata.json")

def load_images(label_to_name: dict, skip_if_celeb_exist=False, max_celebs=MAX_CELEBS):
    docs = load_dataset("tonyassi/celebrity-1000-embeddings", split="train", streaming=True)

    counter = 0
    celebs_counter = 0
    if os.path.exists(metadata_dir):
        metadata_dict = json.load(open(metadata_dir, "r"))
    else:
        metadata_dict = {"celebs": {}}

    metadata = metadata_dict["celebs"]
    celeb_name = None
    prev_celeb_name = None
    skip_curr = False
    for doc in docs:
        img = doc["image"]
        label = doc["label"]
        celeb_name = unidecode(label_to_name[str(label)])
        if prev_celeb_name != celeb_name: # new celeb
            skip_curr = False
            if celebs_counter == max_celebs:
                break
            celebs_counter += 1
            prev_celeb_name = celeb_name
            celeb_dir = os.path.join(celebs_dir, celeb_name)
            if os.path.exists(celeb_dir):
                files = os.listdir(celeb_dir)
                metadata[celeb_name] = {"num_pics": len(files)}
                if skip_if_celeb_exist:
                    print(f"{celeb_name} already exists. Skipping ...")
                    skip_curr = True # skip current celeb in the next iterations
                    continue
            # Create a directory for the celebrity if it doesn't exist
            # if not os.path.exists(celeb_dir):
            else:
                os.makedirs(celeb_dir)
                metadata[celeb_name] = {"num_pics": 0}
            prev_celeb_name = celeb_name
        else: # same celeb as prev doc
            if skip_curr:
                continue

        # Save the image with a unique identifier
        image_path = os.path.join(celeb_dir, f'{celeb_name}_{counter}.jpg')
        img.save(image_path)
        metadata[celeb_name]["num_pics"] += 1
        counter += 1
    print(f"Saved {counter} images of {celebs_counter} celebs")
    json.dump(metadata_dict, open(metadata_dir, "w"))

import re

def extract_names_from_readme(file_path):
    names_dict = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        start_extracting = False
        for line in lines:
            if 'names:' in line:
                start_extracting = True
                continue
            if start_extracting:
                match = re.match(r"\s*'(\d+)':\s*(.*)", line)
                if match:
                    key = int(match.group(1))
                    value = match.group(2).strip()
                    names_dict[key] = value
                else:
                    break
    return names_dict

# Path to the README.md file
file_path = '/home/ubuntu/VecSimDev-POCs/CelebsDemo/README.md'
json_cache_file_path = '/home/ubuntu/VecSimDev-POCs/CelebsDemo/names_cache.json'

# Extract names and convert to dictionary
# names_dict = extract_names_from_readme(file_path)

# Check if the JSON cache file exists
if os.path.exists(json_cache_file_path):
    # Load the dictionary from the JSON cache file
    with open(json_cache_file_path, 'r') as json_file:
        names_dict = json.load(json_file)
else:
    # Extract names and convert to dictionary
    names_dict = extract_names_from_readme(file_path)
    # Save the dictionary to the JSON cache file
    with open(json_cache_file_path, 'w') as json_file:
        json.dump(names_dict, json_file)

# Print the resulting dictionary
# print(names_dict)
if not os.path.exists("Celeb1000"):
    os.mkdir(celebs_dir)
load_images(names_dict, skip_if_celeb_exist=True)

# Read the metadata file and add some statistics
# find the maximum number of images for a celeb
# write it back to the metadata file

def add_general_metadata():
    if os.path.exists(metadata_dir):
        metadata = json.load(open(metadata_dir, "r"))
        celebs_metadata = metadata["celebs"]
        total_num_celebs = len(celebs_metadata)
        num_pics_list = [celebs_metadata[celeb]["num_pics"] for celeb in celebs_metadata]
        total_pics = sum(num_pics_list)
        max_num_pics = max(num_pics_list)
        min_num_pics = min(num_pics_list)
        metadata["statistics"] = {
            "total_num_celebs": total_num_celebs,
            "total_pics": total_pics,
            "max_num_pics": max_num_pics,
            "min_num_pics": min_num_pics
        }
        with open(metadata_dir, "w") as metadata_file:
            json.dump(metadata, metadata_file)
        print(f"Total number of celebs: {total_num_celebs}")
        print(f"Total number of pics: {total_pics}")
        print(f"Max number of images for a celeb: {max_num_pics}")
        print(f"Min number of images for a celeb: {min_num_pics}")

add_general_metadata()
