import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm
import ImageReward as RM
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", type=str, required=True, help="Path to the folder with generated images")
parser.add_argument("--csv_path", type=str, required=True, help="Path to prompts CSV file (CSV with prompt_name, caption)")
parser.add_argument("--output_csv", type=str, default=None, help="Where to save the result CSV")
args = parser.parse_args()

if args.output_csv:
    output_dir = os.path.dirname(args.output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

# Load model
print("Loading ImageReward model...")
model = RM.load("ImageReward-v1.0")
print("Model loaded successfully.")

df = pd.read_csv(args.csv_path)
print(f"Loaded {len(df)} prompts from CSV.")

scores = []

print(f"Scoring images in {args.image_dir} ...")
for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
    prompt_name = row["prompt_name"]
    caption = row["caption"]
    image_path = os.path.join(args.image_dir, prompt_name + ".png")
    
    try:
        if not os.path.exists(image_path):
            scores.append(None)
            continue
            
        try:
            with Image.open(image_path) as pil_img:
                pil_img.verify()
        except Exception as pil_e:
            print(f"Invalid image at {image_path}: {pil_e}")
            scores.append(None)
            continue
        
        score = model.score(caption, [image_path])
        
        if isinstance(score, list) and len(score) > 0:
            score = score[0]
        elif isinstance(score, list) and len(score) == 0:
            score = None
            
        scores.append(score)
        
    except Exception as e:
        print(f"Error at index {i} ({prompt_name}): {e}")
        scores.append(None)

df["image_reward_score"] = scores

if args.output_csv:
    details_csv_path = args.output_csv.replace(".csv", "_details.csv")
    details_dir = os.path.dirname(details_csv_path)
    if details_dir and not os.path.exists(details_dir):
        os.makedirs(details_dir, exist_ok=True)
else:
    details_csv_path = "imagereward_details.csv"

try:
    df.to_csv(details_csv_path, index=False)
    print(f"Detailed results saved to {details_csv_path}")
except Exception as e:
    print(f"Error saving details CSV: {e}")

# Compute average score (excluding None)
valid_scores = df["image_reward_score"].dropna()
if len(valid_scores) > 0:
    final_avg = valid_scores.mean()
    print(f"\nFinal average ImageReward score: {final_avg:.4f}")
    print(f"Valid scores: {len(valid_scores)}/{len(df)} ({len(valid_scores)/len(df)*100:.1f}%)")
    
    summary_df = pd.DataFrame({
        "image_reward_score": [final_avg],
        "valid_count": [len(valid_scores)],
        "total_count": [len(df)]
    })
    
    if args.output_csv:
        try:
            summary_df.to_csv(args.output_csv, index=False)
            print(f"Summary results saved to {args.output_csv}")
        except Exception as e:
            print(f"Error saving summary CSV: {e}")
            # If saving fails, at least write to current directory
            summary_df.to_csv("imagereward_summary.csv", index=False)
            print("Summary results saved to imagereward_summary.csv instead.")
    else:
        summary_df.to_csv("imagereward_summary.csv", index=False)
        print("Summary results saved to imagereward_summary.csv.")
else:
    print("No valid scores computed. Check your images and prompts.")