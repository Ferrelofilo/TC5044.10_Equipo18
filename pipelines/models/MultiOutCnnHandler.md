## MultiOutCnnHandler README

The `MultiOutCnnHandler` class is designed for handling CNN model training, evaluation, and parameter management with PyTorch. It supports logging with MLflow for model tracking and experiment management.

### Requirements

### Models

The handler works with two types of CNN models:
- `SimpleLinearCnnMO3` - A simple linear CNN model with three output heads.
- `ConvolutionalSimpleModel` - A convolutional CNN model with three output heads.

These models should be defined in `pipelines/models/`.

### Usage Example

Below is an end-to-end example showing how to initialize the handler, train, evaluate, save, and load models.

### 1. Initialize the Handler

Create an instance of `MultiOutCnnHandler` with model parameters:

```python
from pipelines.models.multi_output_model_handler import MultiOutCnnHandler

model_params = {
    "input_len": 10,
    "out_features1": 64,
    "out_features2": 32,
    "bias": True
}
cnn_type = "linear_cnn"  # or "convolutional_cnn"

mo_cnn_handler = MultiOutCnnHandler(cnn_type=cnn_type, model_params=model_params)
```

### 2. Train the Model with MLflow Logging

Train the model using a PyTorch DataLoader. The `train_model` method returns a DataFrame with loss and RMSE metrics for each epoch, which can be logged to MLflow. Here’s how to train and log using `mlflow_epochs_logs` and `mlflow_torch_params` helper functions.

```python
train_loader = create_dataloader(X_train, y_train, batch_size=32, shuffle=True)  # Custom function
epochs = 10

# Train the model and get the epoch metrics DataFrame
epoch_df = mo_cnn_handler.train_model(train_loader, epochs=epochs)

# Log training metrics per epoch to MLflow
mlflow_epochs_logs(epoch_df)

# Log additional model parameters to MLflow
ml_params = {"epochs": epochs, "batch_size": 32, "shuffle": True}
mlflow_torch_params(mo_cnn_handler.model, mo_cnn_handler.optimizer, additional_params=ml_params)
```

### 3. Save the Model

Save the trained model, optimizer state, and other configuration parameters to a file:

```python
model_path = "path/to/save/model.pth"
mo_cnn_handler.save_model(model_path)
```

### 4. Load the Model

To evaluate or continue training, load the model along with its configurations:

```python
mo_cnn_handler = MultiOutCnnHandler(cnn_type=cnn_type) # We can add params but it will be replaced in load_model()
mo_cnn_handler.load_model(model_path)
```
When you call load_model, any existing model_params, optimizer_params, or criterion initially set in the handler 
will be replaced by the values loaded from the saved model state. 
This ensures consistency and accurately restores the model's configuration, regardless of any parameters 
passed at initialization.

### 5. Evaluate the Model

To evaluate the model, use the `evaluate_multi_output_metrics` method. This method computes metrics including RMSE, MAE, and R² for each output.

```python
X_test = torch.load(X_test_path)
y_test = torch.load(y_test_path)
test_loader = create_dataloader(X_test, y_test, batch_size=32, shuffle=True)
results_df, total_test_loss = mo_cnn_handler.evaluate_multi_output_metrics(test_loader, mo_cnn_handler.criterion)

# Calculate average test loss
avg_test_loss = sum(total_test_loss) / len(total_test_loss)

# Log evaluation metrics to MLflow
mlflow_evaluate_metrics(results_df, avg_test_loss)

```

### Full Example Workflow with MLflow

Here’s a complete workflow using MLflow:

```python
import mlflow

# Initialize MLflow
mlflow.set_experiment("Example Experiment")
model_type = "linear_cnn"

# Define model parameters and initialize handler
model_params = {
    "input_len": 10,
    "out_features1": 64,
    "out_features2": 32,
    "bias": True
}
mo_cnn_handler = MultiOutCnnHandler(cnn_type=model_type, model_params=model_params)

# Training phase
with mlflow.start_run() as run:
    train_loader = create_dataloader(X_train, y_train, batch_size=32, shuffle=True)
    
    # Train model and log epoch metrics
    epoch_df = mo_cnn_handler.train_model(train_loader, epochs=10)
    mlflow_epochs_logs(epoch_df)

    # Log model parameters
    ml_params = {"epochs": 10, "batch_size": 32, "shuffle": True}
    mlflow_torch_params(mo_cnn_handler.model, mo_cnn_handler.optimizer, additional_params=ml_params)

    # Save the model
    model_path = "path/to/save/model.pth"
    mo_cnn_handler.save_model(model_path)

    # Evaluation phase
    mo_cnn_handler.load_model(model_path)
    test_loader = create_dataloader(X_test, y_test, batch_size=32, shuffle=True)
    results_df, total_test_loss = mo_cnn_handler.evaluate_multi_output_metrics(test_loader, mo_cnn_handler.criterion)
    avg_test_loss = sum(total_test_loss) / len(total_test_loss)
    
    # Log evaluation metrics
    mlflow_evaluate_metrics(results_df, avg_test_loss)

```

---

### Methods Summary

#### `__init__`
Initializes the model handler with specified parameters for model type, model parameters, optimizer, and loss function.

#### `create_model`
Creates the model based on the specified type (`linear_cnn` or `convolutional_cnn`).

#### `train_model`
Trains the model for a specified number of epochs and returns a DataFrame with loss and RMSE metrics.

#### `evaluate_multi_output_metrics`
Evaluates the model on a test dataset and computes metrics for each output, returning a DataFrame with the results.

#### `save_model`
Saves the model, optimizer state, and configuration to a file.

#### `load_model`
Loads the model, optimizer state, and configuration from a file.
