"""
Multi-class Classification using Logistic Regression
===================================================

This module implements multi-class classification using the one-vs-all strategy
with logistic regression. It's designed for educational purposes to demonstrate
how to extend binary logistic regression to handle multiple classes.

The implementation includes:
- Vectorized sigmoid function
- Regularized cost function (excluding intercept)
- Vectorized gradient computation (excluding intercept from regularization)
- One-vs-all training strategy
- Prediction with class probabilities

Author: Refactored for modern Python practices
"""

import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from typing import Tuple, Optional
import warnings

# Suppress specific scipy warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


class MultiClassLogisticRegression:
    """
    Multi-class logistic regression classifier using one-vs-all strategy.

    This classifier trains K binary classifiers for K classes, where each
    classifier distinguishes one class from all others.

    Attributes:
        num_labels (int): Number of classes
        learning_rate (float): Regularization parameter (lambda)
        all_theta (np.ndarray): Trained parameters for all classifiers
    """

    def __init__(self, num_labels: int, learning_rate: float = 1.0):
        """
        Initialize the multi-class logistic regression classifier.

        Args:
            num_labels: Number of classes to classify
            learning_rate: Regularization parameter (lambda)
        """
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.all_theta = None

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Compute sigmoid function.

        Args:
            z: Input array

        Returns:
            Sigmoid of input array
        """
        # Clip z to prevent overflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def cost_function(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute regularized logistic regression cost function.

        The intercept term (theta[0]) is not regularized.

        Args:
            theta: Parameters for the classifier
            X: Feature matrix (m x n+1) with bias column
            y: Binary labels (m x 1)

        Returns:
            Cost value
        """
        m = X.shape[0]

        # Convert to proper shapes for matrix operations
        h = self.sigmoid(X @ theta)

        # Compute cost components
        # Add small epsilon to prevent log(0)
        cost_pos = -y * np.log(h + 1e-15)
        cost_neg = (1 - y) * np.log(1 - h + 1e-15)

        # Regularization term (exclude intercept)
        reg_term = (self.learning_rate / (2 * m)) * np.sum(theta[1:] ** 2)

        return np.sum(cost_pos - cost_neg) / m + reg_term

    def gradient(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute gradient of regularized logistic regression cost function.

        The intercept gradient is not regularized.

        Args:
            theta: Parameters for the classifier
            X: Feature matrix (m x n+1) with bias column
            y: Binary labels (m x 1)

        Returns:
            Gradient vector
        """
        m = X.shape[0]

        # Compute prediction error
        h = self.sigmoid(X @ theta)
        error = h - y.flatten()

        # Compute gradient
        grad = (X.T @ error) / m

        # Add regularization term (exclude intercept)
        grad[1:] += (self.learning_rate / m) * theta[1:]

        return grad

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultiClassLogisticRegression':
        """
        Train the multi-class classifier using one-vs-all strategy.

        Args:
            X: Feature matrix (m x n)
            y: Class labels (m x 1) - labels should be 1-indexed

        Returns:
            self: The fitted classifier
        """
        m, n = X.shape

        # Add bias column
        X_with_bias = np.column_stack([np.ones(m), X])

        # Initialize parameter matrix
        self.all_theta = np.zeros((self.num_labels, n + 1))

        # Train one classifier for each class
        for i in range(1, self.num_labels + 1):
            # Create binary labels for class i vs all others
            y_binary = (y.flatten() == i).astype(int)

            # Initialize parameters for this classifier
            initial_theta = np.zeros(n + 1)

            # Optimize using scipy
            result = minimize(
                fun=self.cost_function,
                x0=initial_theta,
                args=(X_with_bias, y_binary),
                method='TNC',
                jac=self.gradient,
                options={'disp': True}
            )

            # Store optimized parameters
            self.all_theta[i - 1, :] = result.x

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input samples.

        Args:
            X: Feature matrix (m x n)

        Returns:
            Class probabilities (m x num_labels)
        """
        if self.all_theta is None:
            raise ValueError("Model must be fitted before making predictions")

        m = X.shape[0]

        # Add bias column
        X_with_bias = np.column_stack([np.ones(m), X])

        # Compute probabilities for all classes
        probabilities = self.sigmoid(X_with_bias @ self.all_theta.T)

        return probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input samples.

        Args:
            X: Feature matrix (m x n)

        Returns:
            Predicted class labels (m x 1)
        """
        probabilities = self.predict_proba(X)

        # Return class with highest probability (add 1 for 1-indexed labels)
        return np.argmax(probabilities, axis=1) + 1

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score on given data.

        Args:
            X: Feature matrix (m x n)
            y: True class labels (m x 1)

        Returns:
            Accuracy score between 0 and 1
        """
        predictions = self.predict(X)
        return np.mean(predictions == y.flatten())


def load_data(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from MATLAB .mat file.

    Args:
        filename: Path to .mat file

    Returns:
        Tuple of (X, y) where X is features and y is labels
    """
    data = loadmat(filename)
    return data['X'], data['y']


def display_sample_images(X: np.ndarray, y: np.ndarray, num_samples: int = 10):
    """
    Display sample images from the dataset (requires matplotlib).

    Args:
        X: Feature matrix where each row is a flattened 20x20 image
        y: Labels for the images
        num_samples: Number of samples to display
    """
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        axes = axes.ravel()

        # Randomly select samples
        indices = np.random.choice(X.shape[0], num_samples, replace=False)

        for i, idx in enumerate(indices):
            # Reshape flattened image back to 20x20
            image = X[idx].reshape(20, 20)
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f'Label: {y[idx, 0]}')
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib not available. Cannot display images.")


def main():
    """
    Main function demonstrating multi-class classification on handwritten digits.
    """
    print("Multi-class Classification with Logistic Regression")
    print("=" * 50)

    # Load data
    try:
        X, y = load_data('ex3data1.mat')
        print("Data loaded successfully!")
        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
    except FileNotFoundError:
        print("Error: ex3data1.mat file not found!")
        print("Please ensure the data file is in the current directory.")
        return

    # Initialize and train classifier
    print("\nTraining multi-class classifier...")
    classifier = MultiClassLogisticRegression(num_labels=10, learning_rate=1.0)

    # Train the model
    classifier.fit(X, y)
    print("Training completed!")

    # Make predictions
    print("\nEvaluating model performance...")
    predictions = classifier.predict(X)
    accuracy = classifier.score(X, y)

    print(f"Training accuracy: {accuracy:.2%}")

    # Display some predictions
    print("\nSample predictions:")
    for i in range(5):
        idx = np.random.randint(0, X.shape[0])
        pred = predictions[idx]
        true_label = y[idx, 0]
        print(f"Sample {idx}: Predicted = {pred}, True = {true_label}")

    # Display sample images if matplotlib is available
    print("\nDisplaying sample images...")
    display_sample_images(X, y)


if __name__ == "__main__":
    main()
