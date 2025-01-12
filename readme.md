# ML-Augmented Scheduling System

## Overview
The **ML-Augmented Scheduling System** is an advanced task scheduling and priority prediction platform that leverages machine learning and efficient data structures to optimize resource management. The project integrates robust backend methodologies, including threading, priority heaps, and a machine learning regression model, to deliver high-performance scheduling capabilities.

---

## Features
- **Priority Scheduling**: Utilizes a priority heap for efficient task scheduling and execution.
- **Multi-Threading**: Supports concurrent task processing for high performance.
- **Machine Learning Integration**: Predicts task priorities using a pre-trained Random Forest regression model.
- **Reusable Components**: Modularized code for scalability and maintenance.

---

## Key Concepts Used in the Project

### 1. **Threading**
   - The system uses Python’s `threading` library to process multiple tasks concurrently.
   - Each task runs in its own thread, ensuring efficient utilization of system resources and parallel execution.
   - A shared lock mechanism is implemented to maintain thread safety when accessing shared data structures like the priority heap.

### 2. **Priority Heap**
   - The core of the scheduler is a priority heap (implemented using Python’s `heapq` module).
   - Tasks are prioritized based on a calculated priority score, ensuring that higher-priority tasks are executed first.
   - Efficient insertion and extraction from the heap provide optimal performance for scheduling.

### 3. **Machine Learning Model**
   - A Random Forest regressor is trained on task data to predict task priorities.
   - Features like `task_size` and `resource_requirements` are used to calculate a priority score dynamically.
   - The model is integrated seamlessly into the scheduling workflow to enhance decision-making.

### 4. **Argparse Integration**
   - The project uses the `argparse` library to provide a command-line interface for specifying runtime arguments.
   - This allows users to configure task parameters, specify input files, or modify execution behavior directly from the command line.

---

## Project Structure
```plaintext
.
├── Makefile                 # Build automation for common tasks
├── app.py                   # Main application logic
├── assets                   # Static assets
│   └── favicon.ico          # Favicon for the application
├── data                     # Data-related files
│   ├── __init__.py          # Package initialization
│   ├── dataloader.py        # Utilities for loading and managing data
│   └── training_dataset.csv # Dataset used to train the model
├── environment.yml          # Conda environment configuration
├── ml_scheduling            # Main application logic module
│   ├── __init__.py          # Package initialization
│   └── ml_scheduling.py     # Core scheduling logic
├── models                   # Machine learning model files
│   ├── __init__.py          # Package initialization
│   └── model.pkl            # Trained Random Forest model
├── readme.md                # Project documentation
├── requirements.txt         # Python dependencies
├── rxconfig.py              # Reflex configuration file
└── scheduler                # Scheduler-related modules
    ├── __init__.py          # Package initialization
    ├── models_utils.py      # Functions for predictions and model utilities
    └── run_scheduler.py     # Scheduler logic for task management
```

---

## Requirements
- Python 3.9+
- Conda (recommended for environment management)

### Python Libraries
- pandas
- scikit-learn
- joblib
- argparse
- rich

---

## Installation
1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd task-priority-app
   ```

2. **Create a Conda Environment**:
   ```bash
   conda create -n ml-scheduling python=3.9 -y
   conda activate ml-scheduling
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   Use the provided Makefile targets to streamline operations:
   ```bash
   make run_scheduler   # Run the scheduler
   make train_model     # Train the machine learning model
   make predict         # Predict task priority
   ```

---

## Model Training
The priority prediction model is a Random Forest regressor trained on the `training_dataset.csv` dataset. To retrain the model:

1. Modify the dataset in `data/training_dataset.csv` if needed.
2. Use `models/models_utils.py` to retrain the model:
   ```bash
   python models/models_utils.py
   ```
3. The retrained model will be saved to `models/model.pkl`.

---

## Scheduler Design

### Multi-Threading and Task Management
- The scheduler uses a multi-threaded approach where each task is assigned to its own thread. This ensures high throughput and efficient handling of concurrent operations.
- Thread safety is maintained using a locking mechanism to coordinate access to shared resources like the priority heap.

### Priority-Based Execution
- Tasks are added to a priority queue (a max-heap implemented using Python’s `heapq` with negated priorities).
- The scheduler ensures that tasks with higher priority scores are executed before lower-priority tasks.

### Machine Learning Integration
- The model dynamically calculates task priorities based on input features (`task_size`, `resource_requirements`).
- The integration of ML ensures adaptability, allowing the system to optimize scheduling based on historical data.

### Command-Line Interface
- The `argparse` library is used to provide a flexible CLI, allowing users to:
  - Specify task parameters dynamically.
  - Configure execution modes (e.g., training, prediction).
  - Pass file paths for input data or model saving/loading.

---

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any features or fixes.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact
For questions or support, contact:
- **Name**: Kapeesh Kaul
- **Email**: kapeeshkaul@gmail.com

