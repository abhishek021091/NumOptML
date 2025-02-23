import numpy as np
from .base_models import BaseModel

class LinearRegression(BaseModel):

    """
    Linear Regression model using the Normal Equation.
    """
    def __init__(self):
        """
        Inititalizes the Linear Regression model.
        """

        super().__init__()
        self.theta = None # Model parameters (weights)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the Linear Regression model using the Noraml Equation
        
        :param X: Input features, shape (n_samples, n_featues)
        :param y: Target values, shape (n_samples,)
        """
        X = np.c_[np.ones(X.shape[0]),X] #Add bias term (intercept)
        self.theta = np.linalg.pinv(X.T @ X) @ X.T @ y # Compute weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values using the trained model.
        
        :param X: Input features, shape (n_samples, n_features)
        :return: Predicted values, shape (n_samples,)
        """
        X = np.c_[np.ones(X.shape[0]), X] # Add bias term (intercept)
        return X @ self.theta

