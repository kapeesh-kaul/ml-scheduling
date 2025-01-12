import joblib
from pathlib import Path

def load_model(model_path = Path(__file__).parent / "model.pkl"):
    """Load a trained model from a file."""
    return joblib.load(model_path)