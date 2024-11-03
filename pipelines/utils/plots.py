import matplotlib.pyplot as plt


def plot_loss_curve(total_test_loss):
    """
    Grafica la curva de pérdida para el conjunto de prueba, mostrando la pérdida por cada batch.
    Args:
        total_test_loss (list of float): Lista de valores de pérdida para cada batch en el conjunto de prueba.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(total_test_loss, label="Test Loss per Batch")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Test Loss per Batch")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_actual_vs_predicted(y_true, y_pred, label):
    """
    Grafica los valores reales versus los valores predichos para evaluar la precisión del modelo.
    Args:
        y_true (array-like): Valores reales.
        y_pred (array-like): Valores predichos por el modelo.
        label (str): Etiqueta para identificar el gráfico (por ejemplo, tipo de flare).
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color="red", lw=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Actual vs Predicted for {label}")
    plt.grid(True)
    plt.show()