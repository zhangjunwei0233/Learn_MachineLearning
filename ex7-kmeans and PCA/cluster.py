"""
Modern Clustering and Principal Component Analysis Tutorial

A comprehensive, well-structured tutorial covering K-means clustering and PCA
using modern Python practices and scikit-learn implementations.

Author: Refactored from ML-Exercise7.ipynb
Date: 2025
"""

from typing import Tuple, Optional, List, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.io import loadmat
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Handles loading and preprocessing of various datasets."""

    def __init__(self, data_path: str = 'data/'):
        self.data_path = data_path

    def load_2d_dataset(self, filename: str = 'ex7data2.mat') -> np.ndarray:
        """Load 2D clustering dataset."""
        try:
            data = loadmat(f"{self.data_path}{filename}")
            return data['X']
        except FileNotFoundError:
            print(
                f"Data file {filename} not found. Generating synthetic data.")
            return self._generate_synthetic_2d_data()

    def load_pca_dataset(self, filename: str = 'ex7data1.mat') -> np.ndarray:
        """Load 2D PCA dataset."""
        try:
            data = loadmat(f"{self.data_path}{filename}")
            return data['X']
        except FileNotFoundError:
            print(
                f"Data file {filename} not found. Generating synthetic data.")
            return self._generate_synthetic_pca_data()

    def load_face_dataset(self, filename: str = 'ex7faces.mat') -> np.ndarray:
        """Load face images dataset for PCA analysis."""
        try:
            data = loadmat(f"{self.data_path}{filename}")
            return data['X']
        except FileNotFoundError:
            print(
                f"Data file {filename} not found. Generating synthetic face data.")
            return self._generate_synthetic_face_data()

    def load_image_for_compression(self, filename: str = 'bird_small.png') -> np.ndarray:
        """Load image for compression demonstration."""
        try:
            img = plt.imread(f"{self.data_path}{filename}")
            if img.max() > 1:
                img = img / 255.0
            return img
        except FileNotFoundError:
            print(
                f"Image file {filename} not found. Generating synthetic image.")
            return self._generate_synthetic_image()

    def _generate_synthetic_2d_data(self) -> np.ndarray:
        """Generate synthetic 2D clustering data."""
        np.random.seed(42)
        cluster1 = np.random.multivariate_normal(
            [2, 3], [[0.5, 0.1], [0.1, 0.8]], 20)
        cluster2 = np.random.multivariate_normal(
            [6, 3], [[0.8, 0.2], [0.2, 0.5]], 25)
        cluster3 = np.random.multivariate_normal(
            [7, 6], [[0.6, -0.1], [-0.1, 0.7]], 18)
        return np.vstack([cluster1, cluster2, cluster3])

    def _generate_synthetic_pca_data(self) -> np.ndarray:
        """Generate synthetic PCA data."""
        np.random.seed(42)
        n_samples = 50
        # Create correlated 2D data
        cov_matrix = np.array([[2, 1.5], [1.5, 2]])
        return np.random.multivariate_normal([3, 4], cov_matrix, n_samples)

    def _generate_synthetic_face_data(self) -> np.ndarray:
        """Generate synthetic face-like data."""
        np.random.seed(42)
        n_faces, img_size = 100, 1024  # 32x32 images
        return np.random.rand(n_faces, img_size)

    def _generate_synthetic_image(self) -> np.ndarray:
        """Generate synthetic image for compression."""
        np.random.seed(42)
        return np.random.rand(128, 128, 3)


class Visualizer:
    """Handles all visualization tasks for clustering and PCA."""

    def __init__(self):
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def plot_2d_data(self, X: np.ndarray, title: str = "2D Dataset",
                     labels: Optional[np.ndarray] = None, centroids: Optional[np.ndarray] = None,
                     figsize: Tuple[int, int] = (10, 8)) -> None:
        """Plot 2D data with optional cluster labels and centroids."""
        plt.figure(figsize=figsize)

        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

            for i, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(X[mask, 0], X[mask, 1], c=[colors[i]],
                            label=f'Cluster {int(label)}', alpha=0.7, s=60)

            if centroids is not None:
                plt.scatter(centroids[:, 0], centroids[:, 1], c='red',
                            marker='x', s=200, linewidths=3, label='Centroids')
            plt.legend()
        else:
            plt.scatter(X[:, 0], X[:, 1], alpha=0.7, s=60)

        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Feature 1', fontsize=12)
        plt.ylabel('Feature 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_pca_analysis(self, X_original: np.ndarray, X_projected: np.ndarray,
                          X_recovered: np.ndarray, components: np.ndarray) -> None:
        """Plot PCA analysis results."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Original data
        axes[0].scatter(X_original[:, 0], X_original[:, 1], alpha=0.7, s=60)
        axes[0].set_title('Original Data', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Add principal components as arrows
        mean_point = np.mean(X_original, axis=0)
        # Handle both 1D and 2D cases
        n_components_to_show = min(components.shape[0], X_original.shape[1])
        for i in range(n_components_to_show):
            if X_original.shape[1] == 2:
                # For 2D original data
                component = components[i]
                axes[0].arrow(mean_point[0], mean_point[1],
                              component[0]*3, component[1]*3,
                              head_width=0.1, head_length=0.1,
                              fc=f'C{i}', ec=f'C{i}', alpha=0.8,
                              label=f'PC{i+1}')
        axes[0].legend()

        # Projected data
        if X_projected.ndim == 1 or X_projected.shape[1] == 1:
            # 1D projection
            X_proj_flat = X_projected.flatten() if X_projected.ndim > 1 else X_projected
            axes[1].scatter(X_proj_flat, np.zeros_like(
                X_proj_flat), alpha=0.7, s=60)
            axes[1].set_title('Projected Data (1D)',
                              fontsize=14, fontweight='bold')
            axes[1].set_ylabel('0 (Single Dimension)')
        else:
            # 2D projection
            axes[1].scatter(X_projected[:, 0],
                            X_projected[:, 1], alpha=0.7, s=60)
            axes[1].set_title('Projected Data (2D)',
                              fontsize=14, fontweight='bold')
            axes[1].set_xlabel('PC1')
            axes[1].set_ylabel('PC2')
        axes[1].grid(True, alpha=0.3)

        # Recovered data
        axes[2].scatter(X_recovered[:, 0], X_recovered[:, 1],
                        alpha=0.7, s=60, color='orange')
        axes[2].set_title('Recovered Data', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_image_compression(self, original: np.ndarray, compressed_images: List[np.ndarray],
                               k_values: List[int]) -> None:
        """Plot image compression results for different k values."""
        n_images = len(compressed_images) + 1
        fig, axes = plt.subplots(1, n_images, figsize=(4*n_images, 4))

        if n_images == 1:
            axes = [axes]

        axes[0].imshow(original)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        for i, (compressed, k) in enumerate(zip(compressed_images, k_values)):
            axes[i+1].imshow(compressed)
            axes[i+1].set_title(f'K={k} clusters',
                                fontsize=12, fontweight='bold')
            axes[i+1].axis('off')

        plt.tight_layout()
        plt.show()

    def plot_face_images(self, faces: np.ndarray, n_faces: int = 16,
                         title: str = "Face Images") -> None:
        """Plot a grid of face images."""
        if n_faces > faces.shape[0]:
            n_faces = faces.shape[0]

        grid_size = int(np.sqrt(n_faces))
        if grid_size * grid_size < n_faces:
            grid_size += 1

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        axes = axes.flatten() if grid_size > 1 else [axes]

        for i in range(n_faces):
            face_image = faces[i].reshape(32, 32)
            axes[i].imshow(face_image, cmap='gray')
            axes[i].axis('off')

        # Hide remaining subplots
        for i in range(n_faces, len(axes)):
            axes[i].axis('off')

        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_explained_variance(self, pca_model: PCA, n_components: int = 20) -> None:
        """Plot explained variance ratio for PCA components."""
        plt.figure(figsize=(12, 5))

        # Individual explained variance
        plt.subplot(1, 2, 1)
        components = range(
            1, min(n_components + 1, len(pca_model.explained_variance_ratio_) + 1))
        plt.bar(components, pca_model.explained_variance_ratio_[:n_components])
        plt.title('Explained Variance by Component', fontweight='bold')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.grid(True, alpha=0.3)

        # Cumulative explained variance
        plt.subplot(1, 2, 2)
        cumsum = np.cumsum(pca_model.explained_variance_ratio_[:n_components])
        plt.plot(components, cumsum, 'bo-', linewidth=2, markersize=8)
        plt.title('Cumulative Explained Variance', fontweight='bold')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


class KMeansAnalyzer:
    """Modern K-means clustering analysis with scikit-learn."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()

    def find_optimal_k(self, X: np.ndarray, k_range: range = range(2, 11),
                       normalize: bool = True) -> Tuple[List[float], List[float], int]:
        """Find optimal number of clusters using elbow method and silhouette analysis."""
        if normalize:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X

        inertias = []
        silhouette_scores = []

        for k in k_range:
            kmeans = KMeans(
                n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X_scaled)

            inertias.append(kmeans.inertia_)
            if k > 1:  # Silhouette score requires at least 2 clusters
                sil_score = silhouette_score(X_scaled, labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)

        # Find elbow point (simple method)
        optimal_k = self._find_elbow_point(list(k_range), inertias)

        return inertias, silhouette_scores, optimal_k

    def _find_elbow_point(self, k_values: List[int], inertias: List[float]) -> int:
        """Find elbow point using the rate of change method."""
        if len(inertias) < 3:
            return k_values[0]

        # Calculate rate of change
        rates = []
        for i in range(1, len(inertias) - 1):
            rate = (inertias[i-1] - inertias[i+1]) / 2
            rates.append(rate)

        # Find the point where rate of change starts to decrease significantly
        max_rate_idx = np.argmax(rates)
        return k_values[max_rate_idx + 1]

    def fit_predict(self, X: np.ndarray, n_clusters: int, normalize: bool = True) -> np.ndarray:
        """Fit K-means model and return cluster labels."""
        if normalize:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X

        self.model = KMeans(n_clusters=n_clusters,
                            random_state=self.random_state, n_init=10)
        labels = self.model.fit_predict(X_scaled)

        return labels

    def get_centroids(self, original_space: bool = True) -> np.ndarray:
        """Get cluster centroids."""
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit_predict first.")

        centroids = self.model.cluster_centers_

        # if original_space and hasattr(self.scaler, 'mean_'):
        #     # Transform back to original space
        #     centroids = self.scaler.inverse_transform(centroids)

        return centroids

    def analyze_clustering_quality(self, X: np.ndarray, labels: np.ndarray) -> dict:
        """Analyze the quality of clustering results."""
        metrics = {}

        if len(np.unique(labels)) > 1:
            metrics['silhouette_score'] = silhouette_score(X, labels)
            metrics['inertia'] = self.model.inertia_ if self.model else None
        else:
            metrics['silhouette_score'] = 0
            metrics['inertia'] = None

        metrics['n_clusters'] = len(np.unique(labels))

        return metrics


class PCAAnalyzer:
    """Principal Component Analysis with modern scikit-learn implementation."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.pca_model = None
        self.scaler = StandardScaler()

    def fit_transform(self, X: np.ndarray, n_components: Optional[int] = None,
                      normalize: bool = True) -> np.ndarray:
        """Fit PCA and transform data."""
        if normalize:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X

        self.pca_model = PCA(n_components=n_components,
                             random_state=self.random_state)
        X_transformed = self.pca_model.fit_transform(X_scaled)

        return X_transformed

    def inverse_transform(self, X_transformed: np.ndarray, original_space: bool = True) -> np.ndarray:
        """Reconstruct data from PCA space."""
        if self.pca_model is None:
            raise ValueError(
                "PCA model not fitted yet. Call fit_transform first.")

        X_reconstructed = self.pca_model.inverse_transform(X_transformed)

        if original_space and hasattr(self.scaler, 'mean_'):
            # Transform back to original space
            X_reconstructed = self.scaler.inverse_transform(X_reconstructed)

        return X_reconstructed

    def get_components(self) -> np.ndarray:
        """Get principal components."""
        if self.pca_model is None:
            raise ValueError(
                "PCA model not fitted yet. Call fit_transform first.")

        return self.pca_model.components_

    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio for each component."""
        if self.pca_model is None:
            raise ValueError(
                "PCA model not fitted yet. Call fit_transform first.")

        return self.pca_model.explained_variance_ratio_

    def find_optimal_components(self, X: np.ndarray, variance_threshold: float = 0.95) -> int:
        """Find optimal number of components to retain given variance threshold."""
        # Fit PCA with all components first
        temp_pca = PCA()
        if hasattr(self.scaler, 'mean_'):
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = self.scaler.fit_transform(X)

        temp_pca.fit(X_scaled)

        # Find number of components needed for variance threshold
        cumsum_variance = np.cumsum(temp_pca.explained_variance_ratio_)
        optimal_components = np.argmax(
            cumsum_variance >= variance_threshold) + 1

        return optimal_components


class ImageProcessor:
    """Handles image processing tasks using K-means clustering."""

    def __init__(self):
        self.kmeans_analyzer = KMeansAnalyzer()

    def compress_image(self, image: np.ndarray, k: int) -> np.ndarray:
        """Compress image using K-means clustering."""
        original_shape = image.shape

        # Reshape image to 2D array (pixels x channels)
        if len(original_shape) == 3:
            pixels = image.reshape(-1, original_shape[2])
        else:
            pixels = image.reshape(-1, 1)

        # Apply K-means clustering
        labels = self.kmeans_analyzer.fit_predict(pixels, k, normalize=False)
        centroids = self.kmeans_analyzer.get_centroids(original_space=True)

        # Replace each pixel with its cluster centroid
        compressed_pixels = centroids[labels]

        # Reshape back to original image shape
        compressed_image = compressed_pixels.reshape(original_shape)

        # Ensure values are in valid range [0, 1]
        compressed_image = np.clip(compressed_image, 0, 1)

        return compressed_image

    def analyze_compression_quality(self, original: np.ndarray, compressed: np.ndarray) -> dict:
        """Analyze compression quality metrics."""
        mse = np.mean((original - compressed) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')

        return {
            'mse': mse,
            'psnr': psnr,
            'compression_ratio': f"24-bit to {int(np.log2(len(np.unique(compressed.reshape(-1, compressed.shape[-1]), axis=0))))}-bit per channel"
        }


class ClusteringTutorial:
    """Main tutorial orchestration class."""

    def __init__(self, data_path: str = 'data/'):
        self.data_loader = DataLoader(data_path)
        self.visualizer = Visualizer()
        self.kmeans_analyzer = KMeansAnalyzer()
        self.pca_analyzer = PCAAnalyzer()
        self.image_processor = ImageProcessor()

    def demo1_basic_kmeans(self) -> None:
        """Demonstrate basic K-means clustering on 2D data."""
        print("=== Demo 1: Basic K-means Clustering ===\n")

        # Load data
        X = self.data_loader.load_2d_dataset()
        print(f"Loaded dataset with shape: {X.shape}")

        # Visualize original data
        print("Visualizing original data...")
        self.visualizer.plot_2d_data(X, "Original 2D Dataset")

        # Find optimal number of clusters
        print("Finding optimal number of clusters...")
        inertias, sil_scores, optimal_k = self.kmeans_analyzer.find_optimal_k(
            X, range(2, 8))

        print(f"Suggested optimal k: {optimal_k}")

        # Plot elbow curve
        k_values = list(range(2, 8))
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
        plt.title('Elbow Method for Optimal k', fontweight='bold')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Within-cluster Sum of Squares (WCSS)')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(k_values, sil_scores, 'ro-', linewidth=2, markersize=8)
        plt.title('Silhouette Score vs k', fontweight='bold')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Apply K-means with optimal k
        print(f"\nApplying K-means with k={optimal_k}...")
        labels = self.kmeans_analyzer.fit_predict(
            X, optimal_k, normalize=False)
        centroids = self.kmeans_analyzer.get_centroids(original_space=True)

        # Analyze clustering quality
        quality_metrics = self.kmeans_analyzer.analyze_clustering_quality(
            X, labels)
        print(f"Clustering quality metrics: {quality_metrics}")

        # Visualize results
        self.visualizer.plot_2d_data(
            X, f"K-means Results (k={optimal_k})", labels, centroids)

    def demo2_image_compression(self) -> None:
        """Demonstrate image compression using K-means."""
        print("\n=== Demo 2: Image Compression with K-means ===\n")

        # Load image
        image = self.data_loader.load_image_for_compression()
        print(f"Loaded image with shape: {image.shape}")

        # Compress with different k values
        k_values = [4, 8, 16, 32]
        compressed_images = []

        print("Compressing image with different k values...")
        for k in k_values:
            print(f"Compressing with k={k}...")
            compressed = self.image_processor.compress_image(image, k)
            compressed_images.append(compressed)

            # Analyze compression quality
            quality = self.image_processor.analyze_compression_quality(
                image, compressed)
            print(
                f"  k={k}: MSE={quality['mse']:.4f}, PSNR={quality['psnr']:.2f}dB")

        # Visualize results
        self.visualizer.plot_image_compression(
            image, compressed_images, k_values)

    def demo3_pca_2d(self) -> None:
        """Demonstrate PCA on 2D data."""
        print("\n=== Demo 3: PCA on 2D Data ===\n")

        # Load data
        X = self.data_loader.load_pca_dataset()
        print(f"Loaded dataset with shape: {X.shape}")

        # Apply PCA
        print("Applying PCA with 1 component...")
        X_projected = self.pca_analyzer.fit_transform(X, n_components=1)
        X_recovered = self.pca_analyzer.inverse_transform(X_projected)
        components = self.pca_analyzer.get_components()

        # Print PCA results
        explained_var = self.pca_analyzer.get_explained_variance_ratio()
        print(f"Explained variance ratio: {explained_var}")
        print(f"Total explained variance: {np.sum(explained_var):.3f}")

        # Visualize PCA analysis
        self.visualizer.plot_pca_analysis(
            X, X_projected, X_recovered, components)

    def demo4_face_recognition_pca(self) -> None:
        """Demonstrate PCA for face recognition/compression."""
        print("\n=== Demo 4: Face Recognition with PCA ===\n")

        # Load face dataset
        faces = self.data_loader.load_face_dataset()
        print(f"Loaded face dataset with shape: {faces.shape}")

        # Display some original faces
        print("Displaying sample original faces...")
        self.visualizer.plot_face_images(
            faces, n_faces=16, title="Original Faces")

        # Apply PCA with different numbers of components
        components_list = [10, 50, 100, 200]

        for n_components in components_list:
            if n_components >= faces.shape[1]:
                continue

            print(f"\nApplying PCA with {n_components} components...")

            # Transform and recover
            faces_projected = self.pca_analyzer.fit_transform(
                faces, n_components=n_components)
            faces_recovered = self.pca_analyzer.inverse_transform(
                faces_projected)

            # Get explained variance
            explained_var = np.sum(
                self.pca_analyzer.get_explained_variance_ratio())
            print(
                f"Explained variance with {n_components} components: {explained_var:.3f}")

            # Display recovered faces
            self.visualizer.plot_face_images(faces_recovered, n_faces=16,
                                             title=f"Recovered Faces ({n_components} components)")

        # Plot explained variance analysis
        print("\nAnalyzing explained variance...")
        # Refit with more components for analysis
        self.pca_analyzer.fit_transform(
            faces, n_components=min(200, faces.shape[1]))
        self.visualizer.plot_explained_variance(
            self.pca_analyzer.pca_model, n_components=50)

        # Find optimal number of components
        optimal_components = self.pca_analyzer.find_optimal_components(
            faces, variance_threshold=0.95)
        print(f"Components needed for 95% variance: {optimal_components}")

    def demo5_advanced_analysis(self) -> None:
        """Demonstrate advanced clustering and dimensionality reduction."""
        print("\n=== Demo 5: Advanced Analysis ===\n")

        # Load 2D data for comprehensive analysis
        X = self.data_loader.load_2d_dataset()

        # Apply PCA for dimensionality insight
        print("Applying PCA analysis...")
        X_pca = self.pca_analyzer.fit_transform(X, n_components=2)

        print("PCA Results:")
        explained_var = self.pca_analyzer.get_explained_variance_ratio()
        for i, var in enumerate(explained_var):
            print(f"  PC{i+1}: {var:.3f} ({var*100:.1f}%)")

        # Compare clustering on original vs PCA-transformed data
        print("\nComparing clustering performance...")

        # Original data clustering
        labels_orig = self.kmeans_analyzer.fit_predict(
            X, n_clusters=3, normalize=False)
        quality_orig = self.kmeans_analyzer.analyze_clustering_quality(
            X, labels_orig)

        # PCA-transformed data clustering
        kmeans_pca = KMeansAnalyzer()
        labels_pca = kmeans_pca.fit_predict(
            X_pca, n_clusters=3, normalize=False)
        quality_pca = kmeans_pca.analyze_clustering_quality(X_pca, labels_pca)

        print(
            f"Original data silhouette score: {quality_orig['silhouette_score']:.3f}")
        print(
            f"PCA-transformed silhouette score: {quality_pca['silhouette_score']:.3f}")

        # Visualize comparison
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.scatter(X[:, 0], X[:, 1], c=labels_orig, cmap='viridis', alpha=0.7)
        plt.title('Clustering on Original Data', fontweight='bold')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

        plt.subplot(1, 3, 2)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_pca,
                    cmap='viridis', alpha=0.7)
        plt.title('Clustering on PCA-transformed Data', fontweight='bold')
        plt.xlabel('PC1')
        plt.ylabel('PC2')

        plt.subplot(1, 3, 3)
        # Plot agreement between clusterings
        agreement = (labels_orig == labels_pca).astype(int)
        plt.scatter(X[:, 0], X[:, 1], c=agreement, cmap='RdYlBu', alpha=0.7)
        plt.title('Clustering Agreement', fontweight='bold')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar(label='Agreement (1=Same, 0=Different)')

        plt.tight_layout()
        plt.show()

        # Calculate agreement score
        if len(np.unique(labels_orig)) == len(np.unique(labels_pca)):
            agreement_score = adjusted_rand_score(labels_orig, labels_pca)
            print(
                f"Clustering agreement (Adjusted Rand Score): {agreement_score:.3f}")

    def run_complete_tutorial(self) -> None:
        """Run the complete clustering and PCA tutorial."""
        print("🎓 Modern Clustering and PCA Tutorial")
        print("=" * 50)

        # Run all demonstrations
        self.demo1_basic_kmeans()
        self.demo2_image_compression()
        self.demo3_pca_2d()
        self.demo4_face_recognition_pca()
        self.demo5_advanced_analysis()

        print("\n" + "=" * 50)
        print("🎉 Tutorial completed successfully!")
        print("Key takeaways:")
        print("• K-means is effective for well-separated spherical clusters")
        print("• Image compression demonstrates practical ML applications")
        print("• PCA reveals data structure and enables dimensionality reduction")
        print("• Face recognition showcases PCA's power in high-dimensional data")
        print("• Advanced techniques combine multiple ML methods effectively")


def main():
    """Main function to run the tutorial."""
    # Initialize tutorial
    tutorial = ClusteringTutorial()

    # Run specific demo or complete tutorial
    import sys
    if len(sys.argv) > 1:
        demo_name = sys.argv[1].lower()
        if demo_name == 'kmeans':
            tutorial.demo1_basic_kmeans()
        elif demo_name == 'compression':
            tutorial.demo2_image_compression()
        elif demo_name == 'pca2d':
            tutorial.demo3_pca_2d()
        elif demo_name == 'faces':
            tutorial.demo4_face_recognition_pca()
        elif demo_name == 'advanced':
            tutorial.demo5_advanced_analysis()
        else:
            print(
                "Unknown demo. Available demos: kmeans, compression, pca2d, faces, advanced")
    else:
        # Run complete tutorial
        tutorial.run_complete_tutorial()


if __name__ == "__main__":
    main()
