import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
import seaborn as sns
class KANLayer:
    def __init__(self, control_points, spline_degree=3):
        """
        Initializes the KANLayer with the specified control points and spline degree.

        Parameters:
        control_points (array-like): Points where the B-splines are defined.
        spline_degree (int): Degree of the B-splines. Default is 3 (cubic B-splines).
        """
        self.control_points = control_points
        self.spline_degree = spline_degree
        # Randomly initialize the coefficients with a size of len(control_points) + spline_degree - 1
        self.coefficients = np.random.randn(len(control_points) + spline_degree - 1)
        # Generate B-spline basis functions
        self.b_splines = self._create_b_splines()

    def _create_b_splines(self):
        """
        Creates B-spline basis functions using the control points and spline degree.

        Returns:
        b_splines (list): A list of B-spline basis functions.
        """
        # Create the knot vector, which includes 'spline_degree' leading and trailing points for clamping
        knots = np.concatenate(([self.control_points[0]] * self.spline_degree, self.control_points, [self.control_points[-1]] * self.spline_degree))
        # Generate B-spline basis functions
        b_splines = [BSpline.basis_element(knots[i:i + self.spline_degree + 1]) for i in range(len(self.control_points) + self.spline_degree - 1)]
        return b_splines

    def forward(self, input_values):
        """
        Forward pass through the KANLayer.

        Parameters:
        input_values (array-like): The input values where the B-splines are evaluated.

        Returns:
        output_values (array-like): The output of the KANLayer.
        """
        # Evaluate each B-spline basis function at input_values and sum the results weighted by coefficients
        output_values = np.sum([self.coefficients[i] * self.b_splines[i](input_values) for i in range(len(self.b_splines))], axis=0)
        return output_values

    def backward(self, input_values, gradient_output):
        """
        Backward pass through the KANLayer to compute gradients.

        Parameters:
        input_values (array-like): The input values where the B-splines are evaluated.
        gradient_output (array-like): The gradient of the loss with respect to the output of the KANLayer.

        Returns:
        gradients (array-like): The gradients of the loss with respect to the coefficients.
        """
        # Compute the gradient of the loss with respect to each coefficient
        gradients = np.array([np.sum(gradient_output * self.b_splines[i](input_values)) for i in range(len(self.b_splines))])
        return gradients

def mean_squared_error(true_values, predicted_values):
    """
    Computes the mean squared error between the true and predicted values.

    Parameters:
    true_values (array-like): The actual values.
    predicted_values (array-like): The predicted values.

    Returns:
    mse (float): The mean squared error.
    """
    return np.mean((true_values - predicted_values) ** 2)

# Example usage:
control_points = np.linspace(0, 1, 5)  # Define control points
layer = KANLayer(control_points)  # Initialize Kan Layer

input_values = np.linspace(0, 1, 100)  # Input values
true_function = np.sin(2 * np.pi * input_values)  # Define true function

learning_rate = 0.02  # Learning rate
epochs = 1000  # Number of epochs

# Training loop
for epoch in range(epochs):
    predicted_output = layer.forward(input_values)  # Forward pass
    loss = mean_squared_error(true_function, predicted_output)  # Compute loss
    gradient_output = 2 * (predicted_output - true_function) / len(input_values)  # Compute gradient of loss with respect to output

    gradients = layer.backward(input_values, gradient_output)  # Backward pass
    layer.coefficients -= learning_rate * gradients  # Update coefficients

    if epoch % 100 == 0:
        print(f'Epoch {epoch}/{epochs}, Loss: {loss}')

# Plot the output of the KANLayer and the true function using Seaborn

sns.set(style="darkgrid", context="talk")
plt.figure(figsize=(10, 6))
sns.lineplot(x=input_values, y=layer.forward(input_values), label="Predicted Output", linewidth=2.5)
sns.lineplot(x=input_values, y=true_function, label="True Function", linestyle='dashed', linewidth=2.5)
plt.title("KAN_Layer Output vs True Function", fontsize=16)
plt.xlabel("Input Values", fontsize=14)
plt.ylabel("Output Values", fontsize=14)
plt.legend(fontsize=12)
plt.show()