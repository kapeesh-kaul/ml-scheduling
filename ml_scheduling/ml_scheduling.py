import reflex as rx
from rxconfig import config

from models import load_model
from scheduler.models_utils import predict_priority
from scheduler import Task

# Load the pre-trained model
model = load_model()

class State(rx.State):
    """Application state."""
    task_id: int = 1
    task_type: str = "CPU"
    task_size: int = 0
    resource_requirements: int = 0
    predicted_priority: float = 0.0

    def predict(self):
        """Predict the priority for the given task."""
        # Create a Task object
        task_data = {
            "id": self.task_id,
            "task_type": self.task_type,
            "task_size": self.task_size,
            "resource_requirements": self.resource_requirements,
        }
        task = Task(**task_data)

        # Predict priority
        self.predicted_priority = predict_priority(model, task)

def index():
    """The main interface."""
    return rx.center(
        rx.vstack(
            rx.heading("Task Priority Prediction", size="2"),
            rx.text(f"Task ID: {State.task_id}", font_size="md"),
            rx.text("Task Type:", font_size="md"),
            rx.radio_group(
                items=["CPU", "I/O", "Mixed"],
                value=State.task_type,
                on_change=State.set_task_type,
                direction="row",
            ),
            rx.text(f"Task Size: {State.task_size}", font_size="md"),
            rx.slider(
                min_=0,
                max_=100,
                step=1,
                value=[State.task_size],  # Fixed value handling
                on_change=lambda val: State.set_task_size(val[0]),
            ),
            rx.text(f"Resource Requirements: {State.resource_requirements}", font_size="md"),
            rx.slider(
                min_=0,
                max_=20,
                step=1,
                value=[State.resource_requirements],  # Fixed value handling
                on_change=lambda val: State.set_resource_requirements(val[0]),
            ),
            rx.button("Predict Priority", on_click=State.predict),
            rx.text(
                "Predicted Priority: ",
                rx.cond(
                    State.predicted_priority > 0,
                    rx.text(f"{State.predicted_priority:.2f}", font_weight="bold"),
                    rx.text("No prediction yet"),
                ),
            ),
            spacing="2",
        )
    )

# Configure the app
app = rx.App(state=State)
app.add_page(index, title="Task Priority Prediction")
