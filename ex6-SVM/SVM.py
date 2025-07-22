"""
Modern Support Vector Machine (SVM) Teaching Program

This module provides a comprehensive implementation of Support Vector Machines
for educational purposes, demonstrating both linear and non-linear classification.

Author: Modern Python Implementation
License: MIT
"""

from typing import Tuple, Optional, Dict, Any, Union, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore', category=UserWarning,
                        module='multiprocessing')


class DataLoader:
    """Utility class for loading and preprocessing SVM datasets."""

    @staticmethod
    def _shuffle(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Shuffle the dataset randomly.

        Args:
            X: Feature matrix
            y: Labels

        Returns:
            Shuffled (X, y) tuple
        """
        m = X.shape[0]
        indices = np.random.permutation(m)
        X = X[indices]
        y = y[indices]
        return X, y

    @staticmethod
    def load_ex6_data(dataset_num: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load example dataset from MATLAB files.

        Args:
            dataset_num: Dataset number (1, 2, or 3)

        Returns:
            Tuple of (X, y) where X is features and y is labels
        """
        if dataset_num not in [1, 2, 3]:
            raise ValueError("Dataset number must be 1, 2, or 3")

        data_path = f'data/ex6data{dataset_num}.mat'
        raw_data = loadmat(data_path)
        X = raw_data['X']
        y = raw_data['y'].ravel()
        return X, y

    @staticmethod
    def load_ex6_data_split(dataset_num: int, train_ratio: float = 0.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load example dataset with train/test split.

        Args:
            dataset_num: Dataset number (1, 2, or 3)
            train_ratio: Ratio of data to use for training

        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        if dataset_num not in [1, 2, 3]:
            raise ValueError("Dataset number must be 1, 2, or 3")

        data_path = f'data/ex6data{dataset_num}.mat'
        raw_data = loadmat(data_path)
        X = raw_data['X']
        y = raw_data['y'].ravel()
        m_train = int(X.shape[0] * train_ratio)

        # Shuffle data
        X, y = DataLoader._shuffle(X, y)

        # Split data
        X_train = X[:m_train, :]
        y_train = y[:m_train]
        X_test = X[m_train:, :]
        y_test = y[m_train:]

        return X_train, y_train, X_test, y_test

    @staticmethod
    def load_spam_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load spam classification dataset.

        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        train_data = loadmat('data/spamTrain.mat')
        test_data = loadmat('data/spamTest.mat')

        X_train = train_data['X']
        y_train = train_data['y'].ravel()
        X_test = test_data['Xtest']
        y_test = test_data['ytest'].ravel()

        return X_train, y_train, X_test, y_test


class SVMVisualizer:
    """Visualization utilities for SVM demonstrations."""

    @staticmethod
    def plot_2d_data(X: np.ndarray, y: np.ndarray, title: str = "2D Dataset") -> None:
        """Plot 2D dataset with class labels.

        Args:
            X: Feature matrix (n_samples, 2)
            y: Binary labels (0 or 1)
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        positive = X[y == 1]
        negative = X[y == 0]

        ax.scatter(positive[:, 0], positive[:, 1], s=50, marker='x',
                   c='red', label='Positive (Class 1)')
        ax.scatter(negative[:, 0], negative[:, 1], s=50, marker='o',
                   c='blue', label='Negative (Class 0)')

        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.show()

    @staticmethod
    def plot_decision_confidence(X: np.ndarray, y: np.ndarray,
                                 confidence: np.ndarray, title: str) -> None:
        """Plot decision confidence as color-coded scatter plot.

        Args:
            X: Feature matrix
            y: Labels
            confidence: Decision function values
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        scatter = ax.scatter(X[:, 0], X[:, 1], c=confidence,
                             cmap='RdYlBu', s=50, alpha=0.8)

        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(title)
        plt.colorbar(scatter, label='Decision Confidence')
        ax.grid(True, alpha=0.3)
        plt.show()


class GaussianKernel:
    """Custom Gaussian (RBF) kernel implementation."""

    @staticmethod
    def compute(x1: np.ndarray, x2: np.ndarray, sigma: float) -> float:
        """Compute Gaussian kernel between two vectors.

        Args:
            x1: First vector
            x2: Second vector
            sigma: Kernel bandwidth parameter

        Returns:
            Kernel value
        """
        return np.exp(-np.sum((x1 - x2) ** 2) / (2 * sigma ** 2))


class LinearSVMDemo:
    """Linear SVM demonstration with different regularization parameters."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        """Initialize Linear SVM.

        Args:
            C: Regularization parameter
            max_iter: Maximum iterations
        """
        self.C = C
        self.max_iter = max_iter
        self.model = LinearSVC(C=C, loss='hinge', max_iter=max_iter, dual=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearSVMDemo':
        """Fit the linear SVM model.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self for method chaining
        """
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Features to predict

        Returns:
            Predicted labels
        """
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy score.

        Args:
            X: Features
            y: True labels

        Returns:
            Accuracy score
        """
        return self.model.score(X, y)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function values.

        Args:
            X: Features

        Returns:
            Decision function values
        """
        return self.model.decision_function(X)

    @staticmethod
    def optimize_c_parameter(X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray,
                             C_values: Optional[List[float]] = None) -> Tuple[float, List[float], List[float]]:
        """Optimize C parameter using validation set.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            C_values: List of C values to try

        Returns:
            Tuple of (optimal_C, train_scores, test_scores)
        """
        if C_values is None:
            C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]

        train_scores = []
        test_scores = []

        print("Evaluating different C values...")
        print("C Value\t\tTrain Acc\tTest Acc")
        print("-" * 40)

        for C in C_values:
            svm = LinearSVMDemo(C=C).fit(X_train, y_train)
            train_acc = svm.score(X_train, y_train)
            test_acc = svm.score(X_test, y_test)

            train_scores.append(train_acc)
            test_scores.append(test_acc)

            print(f"{C:8.2f}\t\t{train_acc:.4f}\t\t{test_acc:.4f}")

        # Find optimal C and plot
        optimal_C = LinearSVMDemo._plot_c_optimization_curve(
            C_values, train_scores, test_scores)

        return optimal_C, train_scores, test_scores

    @staticmethod
    def _plot_c_optimization_curve(C_values: List[float], train_scores: List[float],
                                   test_scores: List[float]) -> float:
        """Plot C parameter optimization curve.

        Args:
            C_values: List of C parameter values
            train_scores: Training accuracies for each C
            test_scores: Test accuracies for each C

        Returns:
            Optimal C value based on test performance
        """
        plt.figure(figsize=(10, 6))
        plt.semilogx(C_values, train_scores, 'b-o',
                     label='Training Accuracy', markersize=6)
        plt.semilogx(C_values, test_scores, 'r-s',
                     label='Test Accuracy', markersize=6)

        # Find optimal C (highest test score)
        optimal_idx = np.argmax(test_scores)
        optimal_C = C_values[optimal_idx]

        plt.axvline(x=optimal_C, color='green', linestyle='--', alpha=0.7,
                    label=f'Optimal C = {optimal_C}')
        plt.scatter(optimal_C, test_scores[optimal_idx], color='green', s=100,
                    zorder=5, label=f'Best Test Acc = {test_scores[optimal_idx]:.4f}')

        plt.xlabel('C (Regularization Parameter)')
        plt.ylabel('Accuracy')
        plt.title('SVM C Parameter Optimization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(min(C_values) * 0.5, max(C_values) * 2)
        plt.ylim(0.5, 1.05)
        plt.show()

        return optimal_C


class NonLinearSVMDemo:
    """Non-linear SVM demonstration using RBF kernel."""

    def __init__(self, C: float = 1.0, gamma: float = 1.0,
                 kernel: str = 'rbf', probability: bool = False):
        """Initialize Non-linear SVM.

        Args:
            C: Regularization parameter
            gamma: Kernel coefficient
            kernel: Kernel type
            probability: Whether to enable probability estimates
        """
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.probability = probability
        self.model = SVC(C=C, gamma=gamma, kernel=kernel,
                         probability=probability)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NonLinearSVMDemo':
        """Fit the non-linear SVM model.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self for method chaining
        """
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Features to predict

        Returns:
            Predicted labels
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Features

        Returns:
            Class probabilities
        """
        if not self.probability:
            raise ValueError("Probability estimation not enabled")
        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy score.

        Args:
            X: Features
            y: True labels

        Returns:
            Accuracy score
        """
        return self.model.score(X, y)


class SVMHyperparameterTuner:
    """Modern hyperparameter tuning for SVM using GridSearchCV."""

    def __init__(self, cv: int = 5, scoring: str = 'accuracy', n_jobs: int = -1):
        """Initialize hyperparameter tuner.

        Args:
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
        """
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None

    def tune_svm(self, X: np.ndarray, y: np.ndarray,
                 param_grid: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Tune SVM hyperparameters using grid search.

        Args:
            X: Training features
            y: Training labels
            param_grid: Parameter grid for search

        Returns:
            Dictionary containing best parameters and score
        """
        if param_grid is None:
            param_grid = {
                'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
                'gamma': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
            }

        svm_model = SVC(probability=True)  # Enable probability estimation
        grid_search = GridSearchCV(
            svm_model, param_grid, cv=self.cv,
            scoring=self.scoring, n_jobs=self.n_jobs
        )

        grid_search.fit(X, y)

        self.best_estimator_ = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_

        return {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'best_estimator': self.best_estimator_
        }


class SpamClassifier:
    """Spam email classification using SVM."""

    def __init__(self, C: float = 1.0, kernel: str = 'rbf'):
        """Initialize spam classifier.

        Args:
            C: Regularization parameter
            kernel: SVM kernel type
        """
        self.C = C
        self.kernel = kernel
        self.model = SVC(C=C, kernel=kernel)
        self.is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> 'SpamClassifier':
        """Train the spam classifier.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Self for method chaining
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate classifier performance.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with performance metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred)
        }


class SVMAnalyzer:
    """Analysis utilities for SVM performance evaluation."""

    @staticmethod
    def plot_learning_curve_vs_dataset_size(model, X_train: np.ndarray, y_train: np.ndarray,
                                            X_test: np.ndarray, y_test: np.ndarray,
                                            model_type: str = "Linear") -> None:
        """Plot learning curve showing performance vs training set size.

        Args:
            model: Trained SVM model to evaluate
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            model_type: Type of model ("Linear" or "RBF")
        """
        # Generate different training set sizes
        m_total = len(X_train)
        train_sizes = np.linspace(0.1, 1.0, 10)  # 10%, 20%, ..., 100% of data
        m_values = (train_sizes * m_total).astype(int)

        train_errors = []
        test_errors = []

        print(
            f"Evaluating {model_type} SVM performance vs training set size...")
        print("Train Size\tTrain Error\tTest Error")
        print("-" * 40)

        for m in m_values:
            # Train on subset of data
            X_subset = X_train[:m]
            y_subset = y_train[:m]

            # Check if we have both classes in the subset
            unique_classes = np.unique(y_subset)
            if len(unique_classes) < 2:
                # Skip this size if we don't have both classes
                print(f"{m:8d}\t\tSkipped (only one class)")
                continue

            try:
                # Create and train new model with same parameters
                if model_type == "Linear":
                    subset_model = LinearSVMDemo(
                        C=model.C).fit(X_subset, y_subset)
                else:  # RBF
                    from sklearn.svm import SVC
                    probability = hasattr(
                        model, 'predict_proba') and hasattr(model, 'classes_')
                    subset_model = SVC(C=model.C, gamma=model.gamma,
                                       kernel=model.kernel, probability=probability)
                    subset_model.fit(X_subset, y_subset)

                # Calculate errors (1 - accuracy)
                train_error = 1 - subset_model.score(X_subset, y_subset)
                test_error = 1 - subset_model.score(X_test, y_test)

                train_errors.append(train_error)
                test_errors.append(test_error)

                print(f"{m:8d}\t\t{train_error:.4f}\t\t{test_error:.4f}")

            except ValueError as e:
                print(f"{m:8d}\t\tSkipped (error: {str(e)[:30]}...)")
                continue

        # Plot learning curve
        SVMAnalyzer._plot_learning_curve(
            train_errors, test_errors, m_values, model_type)

    @staticmethod
    def _plot_learning_curve(train_errors: List[float], test_errors: List[float],
                             m_values: np.ndarray, model_type: str) -> None:
        """Plot the learning curve visualization.

        Args:
            train_errors: List of training errors
            test_errors: List of test errors
            m_values: Array of training set sizes
            model_type: Type of model for plot title
        """
        # Plot learning curve (only if we have data points)
        if len(train_errors) == 0:
            print("⚠️ No valid data points for plotting - all training sizes had issues")
            return

        # Create corresponding m_values for successful runs
        successful_m_values = m_values[:len(train_errors)]

        plt.figure(figsize=(10, 6))
        plt.plot(successful_m_values, train_errors, 'b-o',
                 label='Training Error', markersize=6)
        plt.plot(successful_m_values, test_errors,
                 'r-s', label='Test Error', markersize=6)

        plt.xlabel('Training Set Size (m)')
        plt.ylabel('Error Rate')
        plt.title(
            f'{model_type} SVM Learning Curve: Performance vs Training Set Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(successful_m_values[0] * 0.9, successful_m_values[-1] * 1.1)

        # Add analysis text
        final_gap = abs(train_errors[-1] - test_errors[-1])
        if final_gap > 0.05:
            bias_variance = "High Variance (Overfitting)"
            color = 'red'
        elif train_errors[-1] > 0.05:
            bias_variance = "High Bias (Underfitting)"
            color = 'orange'
        else:
            bias_variance = "Good Fit"
            color = 'green'

        plt.text(0.02, 0.98, f"Diagnosis: {bias_variance}",
                 transform=plt.gca().transAxes, fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.3",
                           facecolor=color, alpha=0.3),
                 verticalalignment='top')

        plt.tight_layout()
        plt.show()

        # Print analysis
        SVMAnalyzer._print_learning_curve_analysis(
            train_errors, test_errors, final_gap, model_type)

    @staticmethod
    def _print_learning_curve_analysis(train_errors: List[float], test_errors: List[float],
                                       final_gap: float, model_type: str) -> None:
        """Print detailed learning curve analysis.

        Args:
            train_errors: List of training errors
            test_errors: List of test errors
            final_gap: Gap between final train and test errors
            model_type: Type of model for recommendations
        """
        print(f"\n📊 {model_type} SVM Learning Curve Analysis:")
        print(f"Final training error: {train_errors[-1]:.4f}")
        print(f"Final test error: {test_errors[-1]:.4f}")
        print(f"Train-test gap: {final_gap:.4f}")

        if final_gap > 0.05:
            print("🔍 High variance detected - model may benefit from:")
            print("   • More training data")
            print("   • Higher regularization (lower C)")
            if model_type == "RBF":
                print("   • Simpler kernel parameters (lower gamma)")
            else:
                print("   • Feature reduction")
        elif train_errors[-1] > 0.05:
            print("🔍 High bias detected - model may benefit from:")
            print("   • More complex model")
            print("   • Lower regularization (higher C)")
            if model_type == "RBF":
                print("   • More complex kernel parameters (higher gamma)")
            else:
                print("   • More features")
        else:
            print(f"✅ {model_type} SVM appears well-tuned!")


class SVMTutorial:
    """Comprehensive SVM tutorial orchestrating all demonstrations."""

    def __init__(self):
        """Initialize the SVM tutorial."""
        self.data_loader = DataLoader()
        self.visualizer = SVMVisualizer()
        self.analyzer = SVMAnalyzer()

    def run_complete_tutorial(self) -> None:
        """Run the complete SVM tutorial with all demonstrations."""
        print("🤖 Modern SVM Teaching Program")
        print("=" * 50)

        self.demo1_linear_svm_analysis()
        self.demo2_gaussian_kernel()
        self.demo3_nonlinear_svm_analysis()
        self.demo4_advanced_hyperparameter_tuning()
        self.demo5_spam_classification()

        print("\n✅ All demos completed successfully!")

    def demo1_linear_svm_analysis(self) -> None:
        """Demo 1: Linear SVM with dataset splitting, C optimization, and learning curves."""
        print("\n📊 Demo 1: Dataset Splitting and C Parameter Optimization")

        # Load and split data
        X1_train, y1_train, X1_test, y1_test = self.data_loader.load_ex6_data_split(
            1)
        print(f"Training set size: {X1_train.shape}")
        print(f"Test set size: {X1_test.shape}")

        self.visualizer.plot_2d_data(
            X1_train, y1_train, "Dataset 1: Training Data")

        # Quick comparison
        print("\n📈 Quick Comparison: C=1 vs C=100")
        svm_c1 = LinearSVMDemo(C=1).fit(X1_train, y1_train)
        svm_c100 = LinearSVMDemo(C=100).fit(X1_train, y1_train)

        print(f"Linear SVM (C=1) - Train: {svm_c1.score(X1_train, y1_train):.4f}, "
              f"Test: {svm_c1.score(X1_test, y1_test):.4f}")
        print(f"Linear SVM (C=100) - Train: {svm_c100.score(X1_train, y1_train):.4f}, "
              f"Test: {svm_c100.score(X1_test, y1_test):.4f}")

        # C Parameter Optimization
        print("\n⚙️ C Parameter Optimization")
        optimal_C, train_scores, test_scores = LinearSVMDemo.optimize_c_parameter(
            X1_train, y1_train, X1_test, y1_test)

        print(f"\n🎯 Results Summary:")
        print(f"Optimal C value: {optimal_C}")
        print(f"Best test accuracy: {max(test_scores):.4f}")

        # Train final model with optimal C
        print(f"\n🏆 Training Final Model with Optimal C = {optimal_C}")
        final_svm = LinearSVMDemo(C=optimal_C).fit(X1_train, y1_train)
        final_train_acc = final_svm.score(X1_train, y1_train)
        final_test_acc = final_svm.score(X1_test, y1_test)

        print(f"Final Model Performance:")
        print(f"  Training Accuracy: {final_train_acc:.4f}")
        print(f"  Test Accuracy: {final_test_acc:.4f}")

        # Learning Curve Analysis
        print(f"\n📈 Learning Curve vs Dataset Size")
        self.analyzer.plot_learning_curve_vs_dataset_size(
            final_svm, X1_train, y1_train, X1_test, y1_test, "Linear")

        # Decision confidence visualization
        print("\n🔍 Decision Confidence Visualization")
        X1_full, y1_full = self.data_loader.load_ex6_data(1)
        conf1 = svm_c1.decision_function(X1_full)
        conf100 = svm_c100.decision_function(X1_full)

        self.visualizer.plot_decision_confidence(
            X1_full, y1_full, conf1, "SVM Decision Confidence (C=1)")
        self.visualizer.plot_decision_confidence(
            X1_full, y1_full, conf100, "SVM Decision Confidence (C=100)")

    def demo2_gaussian_kernel(self) -> None:
        """Demo 2: Custom Gaussian kernel implementation."""
        print("\n🔧 Demo 2: Custom Gaussian Kernel")
        x1 = np.array([1.0, 2.0, 1.0])
        x2 = np.array([0.0, 4.0, -1.0])
        sigma = 2
        kernel_value = GaussianKernel.compute(x1, x2, sigma)
        print(f"Gaussian kernel value: {kernel_value:.6f}")

    def demo3_nonlinear_svm_analysis(self) -> None:
        """Demo 3: Non-linear SVM with advanced analysis."""
        print("\n🌀 Demo 3: Non-linear SVM with RBF Kernel")

        # Load and split non-linear data
        X2_train, y2_train, X2_test, y2_test = self.data_loader.load_ex6_data_split(
            2)
        print(
            f"Non-linear dataset - Training: {X2_train.shape}, Test: {X2_test.shape}")
        self.visualizer.plot_2d_data(
            X2_train, y2_train, "Dataset 2: Non-linear Training Data")

        # Quick comparison with different C values for RBF
        print("\n📈 Non-linear SVM: Different C Values")
        rbf_c1 = NonLinearSVMDemo(
            C=1, gamma=10, probability=True).fit(X2_train, y2_train)
        rbf_c100 = NonLinearSVMDemo(
            C=100, gamma=10, probability=True).fit(X2_train, y2_train)

        print(f"RBF SVM (C=1) - Train: {rbf_c1.score(X2_train, y2_train):.4f}, "
              f"Test: {rbf_c1.score(X2_test, y2_test):.4f}")
        print(f"RBF SVM (C=100) - Train: {rbf_c100.score(X2_train, y2_train):.4f}, "
              f"Test: {rbf_c100.score(X2_test, y2_test):.4f}")

        # Advanced hyperparameter optimization
        print("\n⚙️ Non-linear SVM: C and Gamma Optimization")
        tuner = SVMHyperparameterTuner()
        results = tuner.tune_svm(X2_train, y2_train)

        print(f"Best parameters: {results['best_params']}")
        print(f"Best cross-validation score: {results['best_score']:.4f}")

        # Train final RBF model with optimal parameters
        optimal_rbf_svm = results['best_estimator']
        optimal_rbf_svm.fit(X2_train, y2_train)

        final_rbf_train_acc = optimal_rbf_svm.score(X2_train, y2_train)
        final_rbf_test_acc = optimal_rbf_svm.score(X2_test, y2_test)

        print(f"\n🏆 Optimal RBF SVM Performance:")
        print(f"  Training Accuracy: {final_rbf_train_acc:.4f}")
        print(f"  Test Accuracy: {final_rbf_test_acc:.4f}")

        # Learning Curve for RBF SVM
        print(f"\n📈 RBF SVM Learning Curve vs Dataset Size")
        self.analyzer.plot_learning_curve_vs_dataset_size(
            optimal_rbf_svm, X2_train, y2_train, X2_test, y2_test, "RBF")

        # Probability visualization
        print("\n🎨 RBF SVM Probability Visualization")
        X2_full, y2_full = self.data_loader.load_ex6_data(2)

        try:
            proba = optimal_rbf_svm.predict_proba(X2_full)[:, 0]
            self.visualizer.plot_decision_confidence(
                X2_full, y2_full, proba, "Optimal RBF SVM Class Probabilities")
        except AttributeError:
            print(
                "⚠️ Probability estimation not available, using decision function instead")
            decision_vals = optimal_rbf_svm.decision_function(X2_full)
            self.visualizer.plot_decision_confidence(
                X2_full, y2_full, decision_vals, "Optimal RBF SVM Decision Function")

    def demo4_advanced_hyperparameter_tuning(self) -> None:
        """Demo 4: Advanced hyperparameter tuning with cross-validation."""
        print("\n⚙️ Demo 4: Advanced Hyperparameter Tuning with Cross-Validation")
        X3, y3 = self.data_loader.load_ex6_data(3)

        tuner = SVMHyperparameterTuner()
        results = tuner.tune_svm(X3, y3)

        print(f"Best parameters: {results['best_params']}")
        print(f"Best cross-validation score: {results['best_score']:.4f}")

    def demo5_spam_classification(self) -> None:
        """Demo 5: Real-world spam email classification."""
        print("\n📧 Demo 5: Spam Email Classification")
        X_train, y_train, X_test, y_test = self.data_loader.load_spam_data()

        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")

        spam_classifier = SpamClassifier().train(X_train, y_train)
        results = spam_classifier.evaluate(X_test, y_test)

        print(f"Spam Classification Accuracy: {results['accuracy']:.4f}")
        print("\nClassification Report:")
        print(results['classification_report'])


def main():
    """Main function using the new class structure."""
    # Create and run the comprehensive SVM tutorial
    tutorial = SVMTutorial()
    tutorial.run_complete_tutorial()


if __name__ == "__main__":
    main()
