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
from pipelines.models.simple_linear_cnn_multi_out_3 import SimpleLinearCnnMO3,evaluate_multi_output_metrics
from pipelines.utils.plots import plot_loss_curve,plot_actual_vs_predicted
from pipelines.utils.data_utils import create_dataloader

def mlflow_evaluate_metrics(results_df):
    # Iterate through the DataFrame and log each metric for all flare categories
    for index, row in results_df.iterrows():
        metric = row["Evaluation Metrics"]
        mlflow.log_metric(f"Test Loss: {test_loss/len(test_loader):.4f}\n")
        mlflow.log_metric(f"RMSE y1 (common flares): {rmse_y1_value:.4f}, MAE y1: {mae_y1:.4f}, R² y1: {r2_y1:.4f}\n")
        mlflow.log_metric(f"RMSE y2 (moderate flares): {rmse_y2_value:.4f}, MAE y2: {mae_y2:.4f}, R² y2: {r2_y2:.4f}\n")
        mlflow.log_metric(f"RMSE y3 (severe flares): {rmse_y3_value:.4f}, MAE y3: {mae_y3:.4f}, R² y3: {r2_y3:.4f}\n")
        # Save the DataFrame as a CSV file and log it as an artifact
    results_df.to_csv("evaluation_metrics.csv", index=False)
    mlflow.log_artifact("evaluation_metrics.csv")

if __name__ == '__main__':
    X_test_path = sys.argv[1]
    y_test_path = sys.argv[2]
    model_path = sys.argv[3]
    report_file_path = sys.argv[4]

    with open(run_id_path,"r") as file:
        run_data = json.load(file)
    run_id = run_data["run_id"]

    model = torch.load(model_path, weights_only=False)
    
    x = torch.load(X_test_path)
    y = torch.load(y_test_path)
    test_loader = create_dataloader(x,y,batch_size=32, shuffle=True)

    # Start the MLflow run
    with mlflow.start_run(run_id=run_id):
        run_id = run.info.run_id
        with open(run_id_out, "w") as f:
            json.dump({"run_id": run_id}, f)
            
        # Add a tag to indicate these are evaluation metrics
        mlflow.set_tag("phase", "evaluation")
        results_df,total_test_loss = evaluate_multi_output_metrics(model, test_loader, criterion, report_file_path)
        mlflow_evaluate_metrics(results_df)
        plot_loss_curve(total_test_loss)
        #Actual vs Predicted Visualizations
        plot_actual_vs_predicted(results_df['y1'], results_df['outputs_y1'], 'Common Flares')
        plot_actual_vs_predicted(results_df['y2'], results_df['outputs_y2'], 'Moderate Flares')
        plot_actual_vs_predicted(results_df['y3'], results_df['outputs_y3'], 'Severe Flares')
