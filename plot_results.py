import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def plot_predictions_and_residuals(true_values, predicted_values, dataset_name = "Test", wandb_name="baseline"):
    """
    Plots True vs. Predicted values and Residuals for each target.

    Parameters:
    - true_values: np.ndarray -> Actual target values (shape: [num_samples, num_targets])
    - predicted_values: np.ndarray -> Model predicted values (shape: [num_samples, num_targets])
    - dataset_name: str -> Name of dataset (e.g., "Validation" or "Test")

    Returns:
    - Displays 2-row subplot per target: (1) True vs. Predicted, (2) Residuals
    """  
    num_targets = true_values.shape[1]
    fig, axes = plt.subplots(2, num_targets, figsize=(5*num_targets, 10))

    for i in range(num_targets):
        true_i = true_values[:, i]
        pred_i = predicted_values[:, i]
        residuals = pred_i - true_i

        # True vs. Predicted
        axes[0, i].scatter(true_i, pred_i, alpha=0.5)
        axes[0, i].plot([true_i.min(), true_i.max()], [true_i.min(), true_i.max()], "r--")  # Identity line
        axes[0, i].set_xlabel("True Values")
        axes[0, i].set_ylabel("Predicted Values")
        axes[0, i].set_title(f"{dataset_name} - Target {i+1}: True vs. Predicted")

        # Residuals plot (Bottom Row)
        axes[1, i].scatter(true_i, residuals, alpha=0.5)
        axes[1, i].axhline(0, color="r", linestyle="--")
        axes[1, i].set_xlabel("True Values")
        axes[1, i].set_ylabel("Residuals (True - Predicted)")
        axes[1, i].set_title(f"{dataset_name} - Target {i+1}: Residuals")
    
    plt.tight_layout()
    plt.savefig(f"results/{dataset_name}_predictions_residuals.png")
    