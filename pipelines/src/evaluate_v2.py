import json
import os
import sys

import mlflow.sklearn

import yaml
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from pipelines.models import MultiOutCnnHandler

from pipelines.utils import (
    mlflow_evaluate_metrics,
    plot_loss_curve,
    plot_actual_vs_predicted,
    create_dataloader,
)

def load_params():
    """
    Carga los parámetros de configuración desde un archivo YAML.
    Returns:
        dict: Diccionario con la configuración cargada desde 'params.yaml'.
    """
    with open("params.yaml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


params = load_params()


if __name__ == "__main__":
    X_test_path = sys.argv[1]
    y_test_path = sys.argv[2]
    model_path = sys.argv[3]
    run_id_path = params["mlflow"]["runs"]
    #report_file_path = sys.argv[5]
    model_type = os.path.basename(model_path).split("_model.pth")[0]  # model_path contiene el nombre del tipo.

    with open(run_id_path, "r") as file:
        run_data = json.load(file)
    run_id = run_data.get(model_type)

    # load test data
    x = torch.load(X_test_path)
    y = torch.load(y_test_path)

    test_loader = create_dataloader(x,y,batch_size=32, shuffle=True)

    cnn = MultiOutCnnHandler(cnn_type=model_type)
    cnn.load_model(model_path)  # Load the saved model state

    with mlflow.start_run(run_id=run_id):

        results_df, total_test_loss = cnn.evaluate_multi_output_metrics(test_loader, cnn.criterion)
        avg_test_loss = sum(total_test_loss) / len(total_test_loss)
        mlflow_evaluate_metrics(results_df, avg_test_loss)
        #plot_loss_curve(total_test_loss)
        #Actual vs Predicted Visualizations
        #plot_actual_vs_predicted(results_df["y1"], results_df["outputs_y1"], "Common Flares")
        #plot_actual_vs_predicted(results_df["y2"], results_df["outputs_y2"], "Moderate Flares")
        #plot_actual_vs_predicted(results_df["y3"], results_df["outputs_y3"], "Severe Flares")
