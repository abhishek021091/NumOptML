import numpy as np

class GradientDescent():
    def __init__(self, learning_rate = 0.01, tol = 1e-6, c1 = 1e6-4, c2 = 0.9):
        self.learning_rate = learning_rate
        self.tol = tol
        self.c1 = c1
        self.c2 = c2

    def optimize(self, X, y):

        """Perform gradient descent optimization with Wolfe conditions."""

        m, n = X.shape
        theta = np.random.randn(n) # Randomly initialize parameter

        while True:
            gradient = (1/m)* X.T @ (X @ theta -y)
            step_size = self._strong_wolfe_line_search(X,y,theta,gradient)
            update = step_size * gradient

            if np.linalg.norm(update) < self.tol: # Convergence check
                break

            theta -= update
        
        return theta
    
    def _strong_wolfe_line_search(self,X,y,theta,gradient):
        """Perform backtracking line search to satisfy Wolfe conditions."""
        step_size = self.learning_rate
        loss = lambda t: np.sum((X @ (theta - t * gradient) - y ) ** 2) /2
        grad_loss = lambda t: np.dot(gradient, (1/X.shape[0]) * X.T @ (X @ (theta - t* gradient) - y))
        
        while loss(step_size) > loss(0) - self.c1 * step_size * np.dot(gradient,gradient) and \
            abs(grad_loss(step_size)) > self.c2 * abs(np.dot(gradient,gradient)):
            step_size *=0.5 #Reduce step size if Armijo condition is not met

        return step_size