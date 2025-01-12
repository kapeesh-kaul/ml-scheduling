import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
from rich import print

def train_model(data, features, target, save_path=None):
    """Train a Random Forest model and save it."""
    X = data[features]
    y = data[target]
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }
    
    # Save the model if a path is provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)
    
    return model, metrics

def predict_priority(model, task):
    """Predict the priority of a single task."""
    # Convert the input task into a DataFrame with feature names
    features = pd.DataFrame(
        [[task.task_size, task.resource_requirements]],
        columns=["task_size", "resource_requirements"]
    )
    return model.predict(features)[0]


if __name__ == "__main__":

    from pathlib import Path
    import sys

    project_dir = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_dir))

    from data.dataloader import load_data
    # Paths
    model_path = Path(__file__).parent.parent / "models" / "model.pkl"
    
    # Load the dataset
    data = load_data()
    
    # Define features and target
    features = ["task_size", "resource_requirements"]
    target = "priority"
    
    # Train the model
    model, metrics = train_model(data, features, target, save_path=model_path)
    
    # Print metrics
    print("Model Training Complete!")
    print(f"MAE: {metrics['MAE']}")
    print(f"MSE: {metrics['MSE']}")
    print(f"R²: {metrics['R2']}")
    print(f"Model saved at: {model_path}")
