import numpy as np
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract base class for all machine learning models in NUMOPTML.
    """
    def __init__(self):
        self.parameters = None # Model parameters (e.g., weights)

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model using the given dataset.
        :param X: Input features, shape (n_samples, n_features)
        :param y: Target values, shape (n_samples,)
        """
        pass
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        :param X: Input featurs, shape (n_samples, n_featurs)
        :return: Predicted values, shape (n_samples,)
        """
        pass

    def score(self, X: np.ndarray, y:np.ndarray) -> float:
        """
        Compute performance metric (e.g., R^2 score).
        :param X: Input features,
        :param y: True target values
        :return: Performance score
        """
        y_pred = self.predict(X)
        return 1-np.sum((y-y_pred) ** 2) /np.sum((y-np.mean(y))**2)