from pathlib import Path
import pandas as pd

def load_data(file_path = Path(__file__).parent / "training_dataset.csv"):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)