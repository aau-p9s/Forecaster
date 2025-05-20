from darts.models import LinearRegressionModel
import os


def load_model_from_path(path):
    if not os.path.exists(path):
        print(f"❌ File does not exist at: {path}")
        return None

    try:
        model = LinearRegressionModel.load(path)
        print(f"✅ Successfully loaded model from: {path}")
        return model
    except EOFError:
        print(
            f"❌ Failed to load model. EOFError: File might be empty or corrupted at {path}"
        )
    except Exception as e:
        print(f"❌ An unexpected error occurred while loading the model: {e}")

    return None


# Example usage
model_path = "./output/daily/LinearRegressionModel/LinearRegressionModel.pth"
model = load_model_from_path(model_path)

if model is not None:
    print("Model is ready to use.")
else:
    print("Model could not be loaded.")
