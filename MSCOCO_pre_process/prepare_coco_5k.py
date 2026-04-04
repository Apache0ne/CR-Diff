import json
import os
import random
import shutil
import csv
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# python MSCOCO_pre_process/prepare_coco_5k.py
#
# Sample 5000 images from original MSCOCO val2014 annotations/images and generate:
#   1) sampled image subset: data/val2014_samples/
#   2) prompts CSV:         data/val2014_samples_prompts.csv
#
# The generated CSV matches CR-Diff evaluation scripts:
#   - columns: prompt_name, caption
#   - prompt_name aligns with saved image file name (without extension)

# Paths (relative to CR-Diff root)
json_path = os.path.join(BASE_DIR, "data/annotations/captions_val2014.json")
image_root = os.path.join(BASE_DIR, "data/val2014")
save_dir = os.path.join(BASE_DIR, "data/val2014_samples")
csv_path = os.path.join(BASE_DIR, "data/val2014_samples_prompts.csv")

# Create output folder
os.makedirs(save_dir, exist_ok=True)

# Read JSON file
with open(json_path, 'r') as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]

# Create image_id -> file_name mapping
id_to_filename = {img["id"]: img["file_name"] for img in images}

# Create image_id -> captions mapping
image_info = defaultdict(lambda: {"file_name": "", "captions": []})
for ann in annotations:
    img_id = ann["image_id"]
    caption = ann["caption"]
    image_info[img_id]["captions"].append(caption)

# Fill in file_name
for img_id in image_info:
    image_info[img_id]["file_name"] = id_to_filename.get(img_id, "UNKNOWN_FILENAME")

# Randomly sample 5000 unique images
random.seed(42)
all_ids = list(image_info.keys())
sampled_ids = random.sample(all_ids, 5000)

# Save CSV
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["prompt_name", "caption"])

    for i, img_id in enumerate(sampled_ids):
        info = image_info[img_id]
        file_name = info["file_name"]
        captions = info["captions"]
        if not os.path.exists(os.path.join(image_root, file_name)):
            print(f"Warning: file not found: {file_name}")
            continue

        caption = random.choice(captions)

        new_name = f"prompt_{i}.jpg"
        src_path = os.path.join(image_root, file_name)
        dst_path = os.path.join(save_dir, new_name)
        shutil.copy(src_path, dst_path)

        writer.writerow([f"prompt_{i}", caption])

print("Completed: generated 5000 sampled images and captions CSV.")
