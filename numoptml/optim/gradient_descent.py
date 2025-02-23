import numpy as np

class GradientDescent():
    def __init__(self, learning_rate = 0.01, tol = 1e-6, c1 = 1e-4, c2 = 0.9, momentum=0.9, nesterov=True, max_step_size = 1.0):
        """
        Universal Gradient Descent Optimizer with Wolfe Conditions, Momentum, and Nesterov Accelaration.
        
        :param learning_rate: Initial step size for gradient descent.
        :param tol: Convergence tolerance for stopping criterion.
        :param c1: Armijo condition constant for Wolfe line search.
        :param c2: Curvature condition constant for Wolfe line search.
        :param momentum: Momentum coefficient (default 0.9).
        :param nesterov: Whether to use Nesterov accalerated gradient (default True)."""
        self.learning_rate = learning_rate
        self.tol = tol
        self.c1 = c1
        self.c2 = c2
        self.momentum = momentum
        self.nesterov = nesterov
        self.max_step_size = max_step_size

    def optimize(self, loss_fn, grad_fn, theta_init):

        """Perform gradient descent optimization with Wolfe conditions.
        
        :param loss_fn: Function that computes the loss.
        :param grad_fn: Function that computes the gradient.
        :param theta_init: Initial parameter values
        :return: Optimized theta
        """

        theta = theta_init
        velocity = np.zeros_like(theta) # Initialize momentum velocity
        prev_loss = float("inf") # Start with a very high loss
        
        while True:
            if self.nesterov:
                lookahead_theta = theta + self.momentum * velocity # Lookahead step
                gradient = grad_fn(lookahead_theta)
            else:
                gradient = grad_fn(theta)

            # Stop when gradient norm is below tolerance (convergence)
            loss = loss_fn(theta)
            if np.linalg.norm(gradient) < self.tol or abs(prev_loss - loss) < self.tol:
                break
            step_size = self._adaptive_wolfe_line_search(loss_fn, grad_fn, theta, gradient)
            velocity = self.momentum * velocity - step_size * gradient # Momentum update
            theta += velocity # Update parameters

            prev_loss = loss
            print(loss)

        return theta
    
    def _adaptive_wolfe_line_search(self,loss_fn, grad_fn, theta, gradient):
        """
        Efficient Wolfe Line Search for Step Size Selection.
        Uses an adaptive approach to prevent excessive shrinking.
        """
        step_size = min(self.learning_rate, self.max_step_size) # Start with the max step
        alpha = 0.5 # Reduction factor for backtracking
        beta = 1.1 # Increase factor for adaptive step-size tuning
        loss_old = loss_fn(theta)
        
        while True:
            new_theta = theta - step_size * gradient
            loss_new = loss_fn(new_theta)
            grad_new = grad_fn(new_theta)

            # Check Wolfe conditions
            armijo = loss_new <= loss_old - self.c1 * step_size * np.dot(gradient, gradient)
            curvature = abs(np.dot(grad_new, gradient)) <= self.c2 * abs(np.dot(gradient, gradient))

            if armijo and curvature:
                return step_size  # Wolfe conditions satisfied

            # If Armijo is violated, decrease step size
            if not armijo:
                step_size *= alpha  # Backtracking

            # If Armijo is satisfied but curvature isn't, try a slightly larger step
            elif armijo and not curvature:
                step_size = min(step_size * beta, self.max_step_size)