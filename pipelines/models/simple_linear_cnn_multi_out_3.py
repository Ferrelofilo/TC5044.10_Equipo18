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
        # Si se agrega esta función en el script de train.py
        # mlflow.log_metric("average_loss", running_loss / len(dataloader), step=epoch + 1)
        # mlflow.log_metric("rmse", epoch_rmse, step=epoch + 1)
        epochs_data.append({
            "epoch": epoch + 1,
            "average_loss": running_loss / len(dataloader),
            "rmse": epoch_rmse
        })
        #print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {running_loss / len(dataloader):.4f}, RMSE: {epoch_rmse:.4f}")

    epochs_df = pd.DataFrame(epochs_data)

    return epochs_df

    # Esto puede correrse después del train para rellenar inmediatamente el mlflow
    # con la respuesta epochs_df
    # for _, row in epochs_df.iterrows():
    #    step = row["epoch"]
    #    mlflow.log_metric("average_loss", row["average_loss"], step=step)
    #    mlflow.log_metric("rmse", row["rmse"], step=step)



