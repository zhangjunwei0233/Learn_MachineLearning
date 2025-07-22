"""
Neural Network with Backpropagation
===================================

This module implements a feedforward neural network with backpropagation
for multi-class classification. It's designed for educational purposes to
demonstrate the key concepts of neural networks including:

- Forward propagation
- Cost function with regularization
- Backpropagation algorithm
- Gradient checking for validation
- Modern Python practices

The implementation follows the structure from Andrew Ng's Machine Learning
course but with improved code organization and documentation.

Author: Refactored for modern Python practices
"""

import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from typing import Tuple, Optional, Dict, Any
import warnings

# Suppress specific scipy warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


class NeuralNetwork:
    """
    A feedforward neural network with one hidden layer for multi-class classification.

    Architecture: input_layer -> hidden_layer -> output_layer

    Attributes:
        input_size (int): Number of input features
        hidden_size (int): Number of hidden units
        output_size (int): Number of output classes
        learning_rate (float): Regularization parameter (lambda)
        theta1 (np.ndarray): Weights for input to hidden layer (hidden_size, input_size + 1)
        theta2 (np.ndarray): Weights for hidden to output layer (output_size, hidden_size + 1)
        trained (bool): Whether the model has been trained
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, learning_rate: float = 1.0):
        """
        Initialize the neural network.

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            output_size: Number of output classes
            learning_rate: Regularization parameter (lambda)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights randomly
        self.theta1 = None
        self.theta2 = None
        self.trained = False

        # Initialize random weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialize weights using random initialization to break symmetry.

        Weights are initialized to small random values in range [-epsilon, epsilon]
        where epsilon is chosen based on the network architecture.
        """
        # Recommended epsilon for weight initialization
        epsilon_init = np.sqrt(6) / np.sqrt(self.input_size + self.hidden_size)

        # Initialize theta1: (hidden_size x input_size+1) - includes bias
        self.theta1 = np.random.uniform(-epsilon_init, epsilon_init,
                                        (self.hidden_size, self.input_size + 1))

        # Initialize theta2: (output_size x hidden_size+1) - includes bias
        self.theta2 = np.random.uniform(-epsilon_init, epsilon_init,
                                        (self.output_size, self.hidden_size + 1))

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Compute sigmoid activation function.

        Args:
            z: Input array

        Returns:
            Sigmoid of input array
        """
        # Clip z to prevent overflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_gradient(z: np.ndarray) -> np.ndarray:
        """
        Compute gradient of sigmoid function.

        Args:
            z: Input array

        Returns:
            Gradient of sigmoid function
        """
        sig = NeuralNetwork.sigmoid(z)
        return sig * (1 - sig)

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """
        Add bias column to the input matrix.

        Args:
            X: Input matrix (m x n)

        Returns:
            Matrix with bias column added (m x n+1)
        """
        m = X.shape[0]
        return np.column_stack([np.ones(m), X])

    def forward_propagate(self, X: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Perform forward propagation through the network.

        Args:
            X: Input data (m x input_size)

        Returns:
            Tuple containing (a1, z2, a2, z3, h) where:
            - a1: Input layer activations with bias (m x input_size+1)
            - z2: Hidden layer pre-activations (m x hidden_size)
            - a2: Hidden layer activations with bias (m x hidden_size+1)
            - z3: Output layer pre-activations (m x output_size)
            - h: Output layer activations (predictions) (m x output_size)
        """
        m = X.shape[0]

        # Layer 1 (input layer) - add bias
        a1 = self._add_bias(X)  # (m x input_size+1)

        # Layer 2 (hidden layer)
        z2 = a1 @ self.theta1.T  # (m x hidden_size)
        a2 = self._add_bias(self.sigmoid(z2))  # (m x hidden_size+1)

        # Layer 3 (output layer)
        z3 = a2 @ self.theta2.T  # (m x output_size)
        h = self.sigmoid(z3)  # (m x output_size)

        return a1, z2, a2, z3, h

    def compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the regularized cost function.

        Args:
            X: Input data (m x input_size)
            y: One-hot encoded labels (m x output_size)

        Returns:
            Cost value
        """
        m = X.shape[0]

        # Forward propagation
        _, _, _, _, h = self.forward_propagate(X)

        # Compute cost
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        cost_positive = -y * np.log(h + epsilon)
        cost_negative = (1 - y) * np.log(1 - h + epsilon)

        J = np.sum(cost_positive - cost_negative) / m

        # Add regularization term (exclude bias terms)
        reg_term = (self.learning_rate / (2 * m)) * (
            np.sum(self.theta1[:, 1:] ** 2) + np.sum(self.theta2[:, 1:] ** 2)
        )

        return J + reg_term

    def backpropagate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Perform backpropagation to compute cost and gradients.

        Args:
            X: Input data (m x input_size)
            y: One-hot encoded labels (m x output_size)

        Returns:
            Tuple of (cost, gradients) where gradients is flattened array
        """
        m = X.shape[0]

        # Forward propagation
        a1, z2, a2, z3, h = self.forward_propagate(X)

        # Compute cost
        cost = self.compute_cost(X, y)

        # Initialize gradient accumulators
        delta1 = np.zeros_like(self.theta1)  # (hidden_size x input_size+1)
        delta2 = np.zeros_like(self.theta2)  # (output_size x hidden_size+1)

        # Backpropagation
        # Output layer error
        d3 = h - y  # (m x output_size)

        # Hidden layer error (Input layer error should be ignored)
        # Remove bias term from theta2 for backprop calculation
        d2_temp = (d3 @ self.theta2[:, 1:]) * \
            self.sigmoid_gradient(z2)  # (m x hidden_size)

        # Accumulate gradients
        delta2 += d3.T @ a2  # (output_size x hidden_size+1)
        delta1 += d2_temp.T @ a1  # (hidden_size x input_size+1)

        # Average gradients
        delta1 /= m
        delta2 /= m

        # Add regularization (exclude bias terms)
        delta1[:, 1:] += (self.learning_rate / m) * self.theta1[:, 1:]
        delta2[:, 1:] += (self.learning_rate / m) * self.theta2[:, 1:]

        # Flatten gradients
        grad = np.concatenate([delta1.ravel(), delta2.ravel()])

        return cost, grad

    def _unpack_params(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unpack flattened parameter array into theta1 and theta2 matrices.

        Args:
            params: Flattened parameter array

        Returns:
            Tuple of (theta1, theta2)
        """
        theta1_size = self.hidden_size * (self.input_size + 1)

        theta1 = params[:theta1_size].reshape(
            self.hidden_size, self.input_size + 1)
        theta2 = params[theta1_size:].reshape(
            self.output_size, self.hidden_size + 1)

        return theta1, theta2

    def _cost_function_wrapper(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        Wrapper function for cost computation during optimization.

        Args:
            params: Flattened parameter array
            X: Input data
            y: One-hot encoded labels

        Returns:
            Cost value
        """
        # Temporarily update weights
        original_theta1, original_theta2 = self.theta1.copy(), self.theta2.copy()
        self.theta1, self.theta2 = self._unpack_params(params)

        cost = self.compute_cost(X, y)

        # Restore original weights
        self.theta1, self.theta2 = original_theta1, original_theta2

        return cost

    def _gradient_function_wrapper(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Wrapper function for gradient computation during optimization.

        Args:
            params: Flattened parameter array
            X: Input data
            y: One-hot encoded labels

        Returns:
            Gradient array
        """
        # Temporarily update weights
        original_theta1, original_theta2 = self.theta1.copy(), self.theta2.copy()
        self.theta1, self.theta2 = self._unpack_params(params)

        _, grad = self.backpropagate(X, y)

        # Restore original weights
        self.theta1, self.theta2 = original_theta1, original_theta2

        return grad

    def fit(self, X: np.ndarray, y: np.ndarray, maxiter: int = 250) -> Dict[str, Any]:
        """
        Train the neural network using backpropagation.

        Args:
            X: Input data (m x input_size)
            y: One-hot encoded labels (m x output_size)
            maxiter: Maximum number of optimization iterations

        Returns:
            Dictionary containing optimization results
        """
        # Pack initial parameters
        initial_params = np.concatenate(
            [self.theta1.ravel(), self.theta2.ravel()])

        # Optimize using scipy
        result = minimize(
            fun=self._cost_function_wrapper,
            x0=initial_params,
            args=(X, y),
            method='TNC',
            jac=self._gradient_function_wrapper,
            options={'maxiter': maxiter}
        )

        # Update weights with optimized parameters
        self.theta1, self.theta2 = self._unpack_params(result.x)
        self.trained = True

        return {
            'success': result.success,
            'cost': result.fun,
            'iterations': result.nit,
            'message': result.message
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input samples.

        Args:
            X: Input data (m x input_size)

        Returns:
            Class probabilities (m x output_size)
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")

        _, _, _, _, h = self.forward_propagate(X)
        return h

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input samples.

        Args:
            X: Input data (m x input_size)

        Returns:
            Predicted class labels (m,) - 1-indexed
        """
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1) + 1  # Convert to 1-indexed

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score on given data.

        Args:
            X: Input data (m x input_size)
            y: True class labels (m,) - can be 1-indexed or one-hot encoded

        Returns:
            Accuracy score between 0 and 1
        """
        predictions = self.predict(X)

        # Handle both 1-indexed labels and one-hot encoded labels
        if y.ndim == 2 and y.shape[1] > 1:  # One-hot encoded
            true_labels = np.argmax(y, axis=1) + 1
        else:  # 1-indexed labels
            true_labels = y.flatten()

        return np.mean(predictions == true_labels)

    def gradient_check(self, X: np.ndarray, y: np.ndarray, epsilon: float = 1e-4) -> float:
        """
        Perform numerical gradient checking to validate backpropagation.

        Args:
            X: Input data (small sample for efficiency)
            y: One-hot encoded labels
            epsilon: Small value for numerical differentiation

        Returns:
            Relative error between numerical and analytical gradients
        """
        # Get analytical gradients
        _, analytical_grad = self.backpropagate(X, y)

        # Compute numerical gradients
        params = np.concatenate([self.theta1.ravel(), self.theta2.ravel()])
        numerical_grad = np.zeros_like(params)

        for i in range(len(params)):
            # Compute f(theta + epsilon)
            params_plus = params.copy()
            params_plus[i] += epsilon
            self.theta1, self.theta2 = self._unpack_params(params_plus)
            cost_plus = self.compute_cost(X, y)

            # Compute f(theta - epsilon)
            params_minus = params.copy()
            params_minus[i] -= epsilon
            self.theta1, self.theta2 = self._unpack_params(params_minus)
            cost_minus = self.compute_cost(X, y)

            # Numerical gradient
            numerical_grad[i] = (cost_plus - cost_minus) / (2 * epsilon)

        # Restore original parameters
        self.theta1, self.theta2 = self._unpack_params(params)

        # Compute relative error
        numerator = np.linalg.norm(numerical_grad - analytical_grad)
        denominator = np.linalg.norm(
            numerical_grad) + np.linalg.norm(analytical_grad)
        relative_error = numerator / denominator

        return relative_error


def one_hot_encode(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert class labels to one-hot encoded format.

    Args:
        y: Class labels (m,) - 1-indexed
        num_classes: Total number of classes

    Returns:
        One-hot encoded labels (m x num_classes)
    """
    m = len(y)
    y_onehot = np.zeros((m, num_classes))
    for i, label in enumerate(y.flatten()):
        y_onehot[i, label - 1] = 1  # Convert from 1-indexed to 0-indexed
    return y_onehot


def shuffle_data(X: np.ndarray, y: np.ndarray, random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shuffle the data randomly to mix up ordered datasets.

    Args:
        X: Feature matrix (m x n)
        y: Labels (m x 1) or (m,)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of shuffled (X, y)
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Create random permutation of indices
    m = X.shape[0]
    indices = np.random.permutation(m)

    # Shuffle both X and y using the same indices
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    return X_shuffled, y_shuffled


def load_data(filename: str, shuffle: bool = True, random_state: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from MATLAB .mat file.

    Args:
        filename: Path to .mat file
        shuffle: Whether to shuffle the data to mix up ordered datasets
        random_state: Random seed for reproducibility (only used if shuffle=True)

    Returns:
        Tuple of (X, y) where X is features and y is labels
    """
    data = loadmat(filename)
    X, y = data['X'], data['y']

    if shuffle:
        X, y = shuffle_data(X, y, random_state)

    return X, y


def display_sample_predictions(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, num_samples: int = 10):
    """
    Display sample predictions with true labels (requires matplotlib).

    Args:
        X: Feature matrix where each row is a flattened 20x20 image
        y_true: True labels
        y_pred: Predicted labels
        num_samples: Number of samples to display
    """
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        axes = axes.ravel()

        # Randomly select samples
        indices = np.random.choice(X.shape[0], num_samples, replace=False)

        for i, idx in enumerate(indices):
            # Reshape flattened image back to 20x20
            image = X[idx].reshape(20, 20)
            axes[i].imshow(image, cmap='gray')

            true_label = y_true[idx] if y_true.ndim == 1 else y_true[idx, 0]
            pred_label = y_pred[idx]

            # Color code: green for correct, red for incorrect
            color = 'green' if true_label == pred_label else 'red'
            axes[i].set_title(
                f'True: {true_label}, Pred: {pred_label}', color=color)
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib not available. Cannot display images.")


def main():
    """
    Main function demonstrating neural network training on handwritten digits.
    """
    print("Neural Network with Backpropagation")
    print("=" * 40)

    # Load data
    try:
        X, y = load_data('ex4data1.mat', shuffle=True)
        print(f"Data loaded successfully!")
        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
    except FileNotFoundError:
        print("Error: ex4data1.mat file not found!")
        print("Please ensure the data file is in the current directory.")
        return

    # Convert labels to one-hot encoding
    num_classes = 10
    y_onehot = one_hot_encode(y, num_classes)
    print(f"One-hot encoded labels shape: {y_onehot.shape}")

    # split it into training set and testing set
    m_train = (X.shape[0] // 10) * 7

    # Initialize neural network
    input_size = X.shape[1]  # 400 features
    hidden_size = 25
    learning_rate = 0.4  # This is the optimized learning rate

    print(f"\nInitializing neural network:")
    print(f"Architecture: {input_size} -> {hidden_size} -> {num_classes}")
    print(f"Regularization parameter: {learning_rate}")

    nn = NeuralNetwork(input_size, hidden_size, num_classes, learning_rate)

    # Gradient checking (on small subset for efficiency)
    print("\nPerforming gradient checking...")
    subset_size = 100
    X_subset = X[:subset_size]
    y_subset = y_onehot[:subset_size]

    relative_error = nn.gradient_check(X_subset, y_subset)
    print(f"Gradient check relative error: {relative_error:.2e}")

    if relative_error < 1e-7:
        print("✓ Gradient check passed! Backpropagation implementation is correct.")
    else:
        print("⚠ Gradient check failed. There may be an error in backpropagation.")

    # Train the network
    print("\nTraining neural network...")
    result = nn.fit(X[:m_train, :], y_onehot[:m_train, :], maxiter=250)

    print("Training completed!")
    print(f"Success: {result['success']}")
    print(f"Final cost: {result['cost']:.6f}")
    print(f"Iterations: {result['iterations']}")

    # Test using training set
    print("\nEvaluating model performance using training set...")
    y_pred_train = nn.predict(X[:m_train, :])
    accuracy_train = nn.score(X[:m_train, :], y[:m_train, :])

    print(f"Training accuracy: {accuracy_train:.2%}")

    print("\nDisplaying sample predictions...")
    display_sample_predictions(X[:m_train, :], y[:m_train, :], y_pred_train)

    # Test using testing set
    print("\nEvaluating model performance using test set...")
    y_pred_test = nn.predict(X[m_train:, :])
    accuracy_test = nn.score(X[m_train:, :], y[m_train:, :])

    print(f"Testing accuracy: {accuracy_test:.2%}")

    print("\nDisplaying sample predictions...")
    display_sample_predictions(X[m_train:, :], y[m_train:, :], y_pred_test)


if __name__ == "__main__":
    main()
