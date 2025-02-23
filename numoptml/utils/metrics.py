import numpy as np

def mean_squared_error(y_true, y_pred):
    """Compute Mean Squared Error (MSE)."""
    return np.mean((y_true-y_pred)**2)

def root_mean_squared_error(y_true, y_pred):
    """Compute Root Mean Squared Error (RMSE)."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_error(y_true, y_pred):
    """Compute Mean Absolute Error (MAE)."""
    return np.mean(np.abs(y_true-y_pred))

def r2_score(y_true, y_pred):
    """Compute R2 Score (Coefficient of Determination)."""
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

def mean_absolute_percentage_error(y_true, y_pred):
    """Compute Mean Absolute Percentage Error (MAPE)."""
    return np.mean(np.abs((y_true - y_pred)/ y_true)) * 100
