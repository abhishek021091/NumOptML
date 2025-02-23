import numpy as np
from .base_models import BaseModel
from numoptml.optim.gradient_descent import GradientDescent
from numoptml.utils.metrics import mean_squared_error, r2_score, mean_absolute_error

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
            self.theta = np.linalg.pinv(X.T @ X) @ X.T @ y # Compute weights using normal equation
        
        elif self.method == "gradient_descent":
            self.theta = self._gradient_descent_optimization(X,y)

    def _gradient_descent_optimization(self, X, y):
        """Helper function to optimize Linear Regression using Gradient Descent."""
        gd = GradientDescent(learning_rate=self.learning_rate, tol=self.tol)

        # Define loss and gradient functions
        loss_fn = lambda theta: np.sum((X @ theta - y) ** 2) / (2 * len(y))
        grad_fn = lambda theta: (1 / len(y)) * X.T @ (X @ theta - y)

        theta_init = np.random.randn(X.shape[1])
        return gd.optimize(loss_fn, grad_fn, theta_init)

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values using the trained model.
        
        :param X: Input features, shape (n_samples, n_features)
        :return: Predicted values, shape (n_samples,)
        """
        X = np.c_[np.ones(X.shape[0]), X] # Add bias term (intercept)
        return X @ self.theta
    
    def evaluate(self, X: np.ndarray, y: np.ndarray):
        """
        Evaluate the model using multiple metrics.
        """
        y_pred = self.predict(X)
        return {
            "MSE": mean_squared_error(y,y_pred),
            "RMSE": np.sqrt(mean_squared_error(y,y_pred)),
            "MAE": mean_absolute_error(y, y_pred),
            "R2 Score": r2_score(y, y_pred),
        }

