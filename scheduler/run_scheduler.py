from pydantic.dataclasses import dataclass
from typing import List
import threading
import time
import heapq
import random
import pandas as pd
from rich import print
from rich.progress import Progress
from pathlib import Path

@dataclass
class Task:
    id: int
    task_size: int
    task_type: str
    resource_requirements: int
    priority: float = None
    execution_time: float = 0.0
    wait_time: float = 0.0

    def run(self):
        """Simulate the execution of the task."""
        print(f"Task {self.id} with priority {self.priority:.2f} is running.")
        time.sleep(self.execution_time)
        print(f"Task {self.id} completed.")

    def __lt__(self, other):
        """Define less-than comparison based on priority for heapq."""
        return self.priority < other.priority


class Scheduler:
    def __init__(self):
        self.task_queue = []  # Priority queue for tasks
        self.lock = threading.Lock()  # Lock for thread safety
        self.dataset = []  # List to store dataset entries

    def calculate_priority(self, task: Task) -> float:
        """Simulate a regression-based priority calculation."""
        # Example: Higher task size and resource requirements result in higher priority
        return task.task_size * 0.6 + task.resource_requirements * 0.4

    def add_task(self, task: Task):
        """Add a task to the priority queue after calculating its priority."""
        with self.lock:
            task.priority = self.calculate_priority(task)
            heapq.heappush(self.task_queue, (-task.priority, task))  # Max-heap with negated priority

    def schedule(self):
        """Run the scheduler to execute tasks based on priority."""
        total_tasks = len(self.task_queue)
        with Progress() as progress:
            task_progress = progress.add_task("[cyan]Processing Tasks", total=total_tasks)
            while self.task_queue:
                with self.lock:
                    _, task = heapq.heappop(self.task_queue)

                # Simulate waiting and execution times
                task.wait_time = random.uniform(0.1, 1)
                task.execution_time = random.uniform(0.5, 2)

                # Log task data for the dataset
                self.dataset.append({
                    "task_id": task.id,
                    "task_size": task.task_size,
                    "task_type": task.task_type,
                    "resource_requirements": task.resource_requirements,
                    "priority": task.priority,
                    "wait_time": task.wait_time,
                    "execution_time": task.execution_time
                })

                task.run()
                progress.update(task_progress, advance=1)


    def get_dataset(self) -> pd.DataFrame:
        """Return the dataset as a Pandas DataFrame."""
        return pd.DataFrame(self.dataset)


if __name__ == "__main__":
    # Example Usage
    scheduler = Scheduler()

    # Simulate task generation
    task_types = ["CPU", "I/O", "Mixed"]
    for i in range(20):
        task = Task(
            id=i,
            task_size=random.randint(10, 100),
            task_type=random.choice(task_types),
            resource_requirements=random.randint(1, 10)
        )
        scheduler.add_task(task)

    # Run scheduler in a separate thread
    scheduler_thread = threading.Thread(target=scheduler.schedule)
    scheduler_thread.start()
    scheduler_thread.join()

    # Retrieve and save the dataset
    dataset = scheduler.get_dataset()
    
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "training_dataset.csv"
    dataset.to_csv(output_path, index=False)

    print(f"Dataset generated and saved as {output_path}")