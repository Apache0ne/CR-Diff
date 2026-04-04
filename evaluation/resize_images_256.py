import os
import argparse
from PIL import Image
from tqdm import tqdm

def resize_png_folder(input_folder, output_folder, target_res=256):
    count = 0
    os.makedirs(output_folder, exist_ok=True)
    png_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(".png")])

    print(f"Found {len(png_files)} PNG files. Starting resize...")

    for file in tqdm(png_files, desc="Resizing", unit="img"):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)
        try:
            with Image.open(input_path) as image:
                image = image.resize((target_res, target_res), Image.BICUBIC)
                image.save(output_path, format="PNG")
                count += 1
        except Exception as e:
            print(f"Failed to process {file}: {e}")

    print(f"Completed resizing to {target_res}px. {count} images resized.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize all PNG images in a folder.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder containing .png images")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder for resized images")
    parser.add_argument("--target_res", type=int, default=256, help="Target resolution (default: 256x256)")

    args = parser.parse_args()
    resize_png_folder(args.input_folder, args.output_folder, args.target_res)
