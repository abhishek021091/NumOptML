import numpy as np
from .base_models import BaseModel
from numoptml.optim.gradient_descent import GradientDescent

class LinearRegression(BaseModel):

    """
    Linear Regression model using the Normal Equation.
    """
    def __init__(self, method = "auto", learning_rate=0.5, tol=1e-8):
        """
        Inititalizes the Linear Regression model.

        :param method: "auto", "normal_eq", or "gradient_descent"
        :param learning_rate = Initial step size for gradient descent
        :param tol: Convergence tolerance for stopping criterion
        """

        super().__init__()
        self.theta = None # Model parameters (weights)
        self.method = method
        self.learning_rate= learning_rate
        self.tol= tol

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the Linear Regression model using the Noraml Equation
        
        :param X: Input features, shape (n_samples, n_featues)
        :param y: Target values, shape (n_samples,)
        """
        X = np.c_[np.ones(X.shape[0]),X] #Add bias term (intercept)

        # Decide method automatically
        if self.method == "auto":
            if X.shape[0] < 10000 and X.shape[1] < 500: # Heuristic condition
                self.method = "normal_eq"
            else:
                self.method = "gradient_descent"

        if self.method == "normal_eq":
            self.theta = np.linalg.pinv(X.T @ X) @ X.T @ y # Compute weights
        elif self.method == "gradient_descent":
            gd = GradientDescent(learning_rate=self.learning_rate,tol=self.tol)

            self.theta = gd.optimize(X,y)

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values using the trained model.
        
        :param X: Input features, shape (n_samples, n_features)
        :return: Predicted values, shape (n_samples,)
        """
        X = np.c_[np.ones(X.shape[0]), X] # Add bias term (intercept)
        return X @ self.theta

