import torch


class RMSEMetric:
    """
    Clase que calcula el RMSE (Root Mean Squared Error) para múltiples salidas (common, moderate, severe flares).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Reinicia los valores acumulados para calcular el RMSE en cada época.
        """
        self.sum_squared_errors_y1 = 0.0
        self.sum_squared_errors_y2 = 0.0
        self.sum_squared_errors_y3 = 0.0
        self.num_samples = 0

    def update(self, output, target, index):
        """
        Actualiza los errores acumulados para cada salida objetivo.
        """
        squared_error = (output - target) ** 2
        if index == 0:
            self.sum_squared_errors_y1 += squared_error.sum().item()
        elif index == 1:
            self.sum_squared_errors_y2 += squared_error.sum().item()
        elif index == 2:
            self.sum_squared_errors_y3 += squared_error.sum().item()
        self.num_samples += target.size(0)

    def compute(self):
        """
        Calcula el RMSE promedio para cada salida (common, moderate, severe flares).
        """
        rmse_y1 = torch.sqrt(
            torch.tensor(self.sum_squared_errors_y1 / self.num_samples)
        )
        rmse_y2 = torch.sqrt(
            torch.tensor(self.sum_squared_errors_y2 / self.num_samples)
        )
        rmse_y3 = torch.sqrt(
            torch.tensor(self.sum_squared_errors_y3 / self.num_samples)
        )

        return rmse_y1, rmse_y2, rmse_y3
