import os
import pickle
import cloudpickle
import torch

# Base input directory
input_base_dir = os.path.expanduser("~/p10/Forecaster/output/hourly/experiment1")
output_base_dir = os.path.join(input_base_dir, "cloudpickle")

# Create base output dir
os.makedirs(output_base_dir, exist_ok=True)

# Walk through subdirectories
for root, dirs, files in os.walk(input_base_dir):
    for file in files:
        if file.endswith(".pth"):
            input_path = os.path.join(root, file)
            print(f"Considering file {file}")

            # Compute relative subpath from base dir and mirror it in output dir
            rel_path = os.path.relpath(input_path, input_base_dir)
            output_path = os.path.join(output_base_dir, rel_path)

            # Ensure the output subdirectory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            try:
                print("Loading...")
                try:
                    obj = torch.load(input_path, map_location="cpu")
                except Exception:
                    # If it fails, fallback to normal pickle
                    with open(input_path, "rb") as f:
                        obj = pickle.load(f)

                print("Saving...")
                # Save with cloudpickle
                with open(output_path, "wb") as f:
                    cloudpickle.dump(obj, f)

                print("Saved")
                print(f"✅ Converted: {rel_path}")
            except Exception as e:
                print(f"❌ Failed to process {rel_path}: {e}")
