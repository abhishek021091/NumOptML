import numpy as np

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Squared Error (MSE).
    
    :param y_true: Actual target values.
    :param y_pred: Predicted target values.
    :return: Mean Squared Error as a float.
    """
    return np.mean((y_true-y_pred)**2)

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error (RMSE).

    :param y_true: Actual target values.
    :param y_pred: Predicted target values.
    :return: Root Mean Squared Error as a float.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error (MAE).
    
    :param y_true: Actual target values.
    :param y_pred: Predicted target values.
    :return: Mean Absolute Error as a float.
    """
    return np.mean(np.abs(y_true-y_pred))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R2 Score (Coefficient of Determination).
    :param y_true: Actual target values.
    :param y_pred: Predicted target values.
    :return: R2 Score as a float.
    """
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.array) -> float:
    """Compute Mean Absolute Percentage Error (MAPE).
    
    :param y_true: Actual target values.
    :param y_pred: Predicted target values.
    :return: Mean Absolute Percentage Error as a float.
    """
    return np.mean(np.abs((y_true - y_pred)/ y_true)) * 100
