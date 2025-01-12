from models import load_model
from scheduler.models_utils import predict_priority
from scheduler import Task
from rich import print

if __name__ == "__main__":
    task_data = {
        'id': 1,
        'task_type': 'CPU',
        'task_size': 45,
        'resource_requirements': 5
    }
    task = Task(**task_data)
    model = load_model()
    priority = predict_priority(model, task)
    print(f"Predicted Priority for Task {task.id}: {priority:.2f}")



