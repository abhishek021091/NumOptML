import numpy as np

class GradientDescent():
    def __init__(self, learning_rate = 0.01, tol = 1e-6, c1 = 1e6-4, c2 = 0.9):
        self.learning_rate = learning_rate
        self.tol = tol
        self.c1 = c1
        self.c2 = c2

    def optimize(self, loss_fn, grad_fn, theta_init):

        """Perform gradient descent optimization with Wolfe conditions.
        
        :param loss_fn: Function that computes the loss.
        :param grad_fn: Function that computes the gradient.
        :param theta_init: Initial parameter values
        
        :return: Optimized theta"""

        theta = theta_init
        
        while True:
            gradient = grad_fn(theta)

            # Stop when gradient norm is below tolerance (convergence)
            if np.linalg.norm(gradient) < self.tol:
                break
            step_size = self._strong_wolfe_line_search(loss_fn, grad_fn, theta, gradient)
            theta -= step_size* gradient

        return theta
    
    def _strong_wolfe_line_search(self,loss_fn, grad_fn, theta, gradient):
        """Perform backtracking line search to satisfy Wolfe conditions."""
        step_size = self.learning_rate
        
        while loss_fn(theta-step_size*gradient) > loss_fn(theta) - self.c1 * step_size * np.dot(gradient, gradient) and \
            abs(np.dot(grad_fn(theta - step_size* gradient), gradient)) > self.c2 * abs(np.dot(gradient, gradient)):
            step_size *= 0.5 # Reduce step size if Wolfe conditions are not met
            
        return step_size