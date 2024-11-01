import pandas as pd
import torch
import torch.nn as nn


class SimpleLinearCnnMO3(nn.Module):
    def __init__(self, input_len=10, out_features1=64,out_features2=32, bias=True):
        super(SimpleLinearCnnMO3, self).__init__()
        self.first_dense = nn.Linear(input_len, out_features1, bias=bias)
        self.second_dense = nn.Linear(out_features1, out_features2, bias=bias)
        self.y1_output = nn.Linear(out_features2, 1)
        self.y2_output = nn.Linear(out_features2, 1)
        self.y3_output = nn.Linear(out_features2, 1)

    def forward(self, x):
        x = torch.relu(self.first_dense(x))
        x = torch.relu(self.second_dense(x))
        y1 = self.y1_output(x)  # common_flares
        y2 = self.y2_output(x)  # moderate_flares
        y3 = self.y3_output(x)  # severe_flares
        return y1, y2, y3

class ConvolutionalSimpleModel(nn.Module):
    def __init__(self, input_len=10, out_features1=64, out_features2=32, kernel_size=3, padding=1):
        super(ConvolutionalSimpleModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=out_features1, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(in_channels=out_features1, out_channels=out_features2, kernel_size=kernel_size, padding=padding)
        #Could add batch normalization
        self.flatten = nn.Flatten()
        self.y1_output = nn.Linear(out_features2 * input_len, 1)  # Adjusting input size
        self.y2_output = nn.Linear(out_features2 * input_len, 1)
        self.y3_output = nn.Linear(out_features2 * input_len, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        y1 = self.y1_output(x)  # common_flares
        y2 = self.y2_output(x)  # moderate_flares
        y3 = self.y3_output(x)  # severe_flares
        return y1, y2, y3

# Esta función puede ir en train o podemos usar el resultado de df para rellenar mlflow usando el step=
# train _v2 usara este acercamiento
def train_model(model, dataloader, optimizer, criterion, rmse_metric, epochs=10):
    model.train()
    epochs_data = []
    for epoch in range(epochs):
        running_loss = 0.0

        rmse_metric.reset()

        for batch, y1, y2, y3 in dataloader:
            optimizer.zero_grad()

            outputs_y1, outputs_y2, outputs_y3 = model(batch)

            loss_y1 = criterion(outputs_y1, y1)
            loss_y2 = criterion(outputs_y2, y2)
            loss_y3 = criterion(outputs_y3, y3)

            loss = loss_y1 + loss_y2 + loss_y3

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            rmse_metric.update(outputs_y1, y1)
            rmse_metric.update(outputs_y2, y2)
            rmse_metric.update(outputs_y3, y3)

        epoch_rmse = rmse_metric.compute().item()
        epochs_data.append({
            "epoch": epoch + 1,
            "average_loss": running_loss / len(dataloader),
            "rmse": epoch_rmse
        })
    epochs_df = pd.DataFrame(epochs_data)

    return epochs_df

    # Esto puede correrse después del train para rellenar inmediatamente el mlflow
    # con la respuesta epochs_df
    # for _, row in epochs_df.iterrows():
    #    step = row["epoch"]
    #    mlflow.log_metric("average_loss", row["average_loss"], step=step)
    #    mlflow.log_metric("rmse", row["rmse"], step=step)

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
            "Common Flares (y1)": [rmse_y1_value, mae_y1, r2_y1,y1,outputs_y1],
            "Moderate Flares (y2)": [rmse_y2_value, mae_y2, r2_y2,y2,outputs_y2],
            "Severe Flares (y3)": [rmse_y3_value, mae_y3, r2_y3,y3,outputs_y3]
        }
    )

    rmse_y1.reset()
    rmse_y2.reset()
    rmse_y3.reset()

    return results_df,total_test_loss

