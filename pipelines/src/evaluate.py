import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
#!pip install torch
import torch
import torchmetrics
from sklearn.metrics import mean_absolute_error, r2_score

def evaluate_multi_output_metrics(model, test_loader, criterion):
    model.eval()

    rmse_y1 = torchmetrics.MeanSquaredError(squared=False)
    rmse_y2 = torchmetrics.MeanSquaredError(squared=False)
    rmse_y3 = torchmetrics.MeanSquaredError(squared=False)

    test_loss = 0.0
    total_test_loss = []

    all_y1_true, all_y2_true, all_y3_true = [], [], []
    all_y1_pred, all_y2_pred, all_y3_pred = [], [], []

    with torch.no_grad():
        for inputs, y1, y2, y3 in test_loader:

            outputs_y1, outputs_y2, outputs_y3 = model(inputs)

            loss_y1 = criterion(outputs_y1, y1)
            loss_y2 = criterion(outputs_y2, y2)
            loss_y3 = criterion(outputs_y3, y3)

            loss = loss_y1 + loss_y2 + loss_y3
            test_loss += loss.item()
            total_test_loss.append(loss.item())

            rmse_y1.update(outputs_y1, y1)
            rmse_y2.update(outputs_y2, y2)
            rmse_y3.update(outputs_y3, y3)

            all_y1_true.extend(y1.cpu().numpy())
            all_y2_true.extend(y2.cpu().numpy())
            all_y3_true.extend(y3.cpu().numpy())

            all_y1_pred.extend(outputs_y1.cpu().numpy())
            all_y2_pred.extend(outputs_y2.cpu().numpy())
            all_y3_pred.extend(outputs_y3.cpu().numpy())

    rmse_y1_value = rmse_y1.compute().item()
    rmse_y2_value = rmse_y2.compute().item()
    rmse_y3_value = rmse_y3.compute().item()

    # numpy arrays para MAE y R²
    all_y1_true = np.array(all_y1_true)
    all_y2_true = np.array(all_y2_true)
    all_y3_true = np.array(all_y3_true)

    all_y1_pred = np.array(all_y1_pred)
    all_y2_pred = np.array(all_y2_pred)
    all_y3_pred = np.array(all_y3_pred)

    # Calculando MAE and R² for each output
    mae_y1 = mean_absolute_error(all_y1_true, all_y1_pred)
    mae_y2 = mean_absolute_error(all_y2_true, all_y2_pred)
    mae_y3 = mean_absolute_error(all_y3_true, all_y3_pred)

    r2_y1 = r2_score(all_y1_true, all_y1_pred)
    r2_y2 = r2_score(all_y2_true, all_y2_pred)
    r2_y3 = r2_score(all_y3_true, all_y3_pred)

    results_df = pd.DataFrame(
        {
            "Metric": ["RMSE", "MAE", "R²"],
            "Common Flares (y1)": [rmse_y1_value, mae_y1, r2_y1],
            "Moderate Flares (y2)": [rmse_y2_value, mae_y2, r2_y2],
            "Severe Flares (y3)": [rmse_y3_value, mae_y3, r2_y3],
        }
    )

    write_regression_report('TC5044.10_Equipo18\\pipelines\\reporting\\regression_report.txt', test_loss, test_loader, rmse_y1_value, 
                            mae_y1, r2_y1,rmse_y2_value, mae_y2, r2_y2,rmse_y3_value, mae_y3, r2_y3)

    #Loss Curve Visualization
    plot_loss_curve(total_test_loss)

    #Actual vs Predicted Visualizations
    plot_actual_vs_predicted(y1, outputs_y1, 'Common Flares')
    plot_actual_vs_predicted(y2, outputs_y2, 'Moderate Flares')
    plot_actual_vs_predicted(y3, outputs_y3, 'Severe Flares')

    rmse_y1.reset()
    rmse_y2.reset()
    rmse_y3.reset()

    return results_df

def write_regression_report(file_path, test_loss, test_loader, rmse_y1_value, mae_y1, r2_y1, 
                            rmse_y2_value, mae_y2, r2_y2, rmse_y3_value, mae_y3, r2_y3):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        # Test loss
        f.write(f"Test Loss: {test_loss/len(test_loader):.4f}\n")
        # Metrics for y1 (common flares)
        f.write(f"RMSE y1 (common flares): {rmse_y1_value:.4f}, MAE y1: {mae_y1:.4f}, R² y1: {r2_y1:.4f}\n")
        # Metrics for y2 (moderate flares)
        f.write(f"RMSE y2 (moderate flares): {rmse_y2_value:.4f}, MAE y2: {mae_y2:.4f}, R² y2: {r2_y2:.4f}\n")
        # Metrics for y3 (severe flares)
        f.write(f"RMSE y3 (severe flares): {rmse_y3_value:.4f}, MAE y3: {mae_y3:.4f}, R² y3: {r2_y3:.4f}\n")

def plot_loss_curve(test_loss_history):
    plt.figure(figsize=(8, 6))
    plt.plot(test_loss_history, label='Test Loss per Batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Test Loss per Batch')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_actual_vs_predicted(y_true, y_pred, label):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted for {label}')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    model = sys.argv[1]
    test_loader = sys.argv[2]
    criterion = sys.argv[3]
    evaluate_multi_output_metrics(model, test_loader, criterion)
