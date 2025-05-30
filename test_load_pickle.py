import os
import cloudpickle

cloudpickle_dir = os.path.expanduser(
    "~/p10/Forecaster/output/hourly/experiment1/cloudpickle"
)


def validate_cloudpickle_models(directory):
    success_count = 0
    failure_count = 0

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".pth"):
                path = os.path.join(root, file)
                try:
                    with open(path, "rb") as f:
                        _ = cloudpickle.load(f)
                    print(f"✅ Loaded successfully: {os.path.relpath(path, directory)}")
                    success_count += 1
                except Exception as e:
                    print(f"❌ Failed to load {os.path.relpath(path, directory)}: {e}")
                    failure_count += 1

    print("\nSummary:")
    print(f"  ✅ Successfully loaded: {success_count}")
    print(f"  ❌ Failed to load:      {failure_count}")


if __name__ == "__main__":
    validate_cloudpickle_models(cloudpickle_dir)

