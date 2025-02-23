import numpy as np
import pytest
from numoptml.models.linear_regression import LinearRegression

def test_linear_regression_fit():
    """Test if Linear Regression correctly fits data."""
    X = np.array([[1],[2],[3],[4],[5]]) # Simple Input
    y = np.array([2,4,6,8,10])

    for method in ["normal_eq", "gradient_descent"]:
        model = LinearRegression(method=method)
        model.fit(X, y)

        # The model should learn weights close to [2] and intercept close to 0
        np.testing.assert_almost_equal(model.theta[0], 0, decimal=5)
        np.testing.assert_almost_equal(model.theta[1], 2, decimal=5)

def test_linear_regression_predict():
    """Test if Linear Regression makes correct predictions."""
    X=np.array([[1],[2],[3],[4],[5]])
    y = np.array([2,4,6,8,10])

    for method in ["normal_eq", "gradient_descent"]:
        model = LinearRegression(method=method)
        model.fit(X,y)
        predictions = model.predict(X)

        # Predictions should be very close to actual values
        np.testing.assert_almost_equal(predictions, y, decimal=5)

def test_linear_regression_score():
    """Test if R2 score is 1.0 for perfect data."""
    X = np.array([[1],[2],[3],[4],[5]])
    y = np.array([2,4,6,8,10])

    for method in ["normal_eq", "gradient_descent"]:
        model = LinearRegression(method=method)
        model.fit(X,y)
        score = model.score(X,y)

        assert score == pytest.approx(1.0, rel = 1e-5) #R2 should be ~1 for perfect fit

if __name__ == "__main__":
    pytest.main(["-v", "--tb=short"])