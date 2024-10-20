import matplotlib.pyplot as plt

def plot_loss_curve(total_test_loss):
    plt.figure(figsize=(8, 6))
    plt.plot(total_test_loss, label='Test Loss per Batch')
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