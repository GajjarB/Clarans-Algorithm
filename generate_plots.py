
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Set
import time

# --- Helper Functions from Notebook ---

def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    return np.sqrt(np.sum((point1 - point2) ** 2))

def calculate_total_cost(data: np.ndarray, medoids: np.ndarray, medoid_indices: List[int]) -> float:
    total_cost = 0.0
    n_samples = data.shape[0]
    for i in range(n_samples):
        if i in medoid_indices:
            continue
        min_distance = float('inf')
        for medoid in medoids:
            distance = euclidean_distance(data[i], medoid)
            if distance < min_distance:
                min_distance = distance
        total_cost += min_distance
    return total_cost

def assign_clusters(data: np.ndarray, medoids: np.ndarray) -> np.ndarray:
    n_samples = data.shape[0]
    labels = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        min_distance = float('inf')
        best_cluster = 0
        for j, medoid in enumerate(medoids):
            distance = euclidean_distance(data[i], medoid)
            if distance < min_distance:
                min_distance = distance
                best_cluster = j
        labels[i] = best_cluster
    return labels

# --- CLARANS Implementation ---

class CLARANS:
    def __init__(self, n_clusters: int, numlocal: int = 2, maxneighbor: int = None, random_state: int = None):
        self.n_clusters = n_clusters
        self.numlocal = numlocal
        self.maxneighbor = maxneighbor
        self.random_state = random_state
        self.best_medoids_ = None
        self.best_medoid_indices_ = None
        self.labels_ = None
        self.cost_ = None
        self.n_iter_ = 0
    
    def fit(self, X: np.ndarray) -> 'CLARANS':
        if self.random_state is not None:
            np.random.seed(self.random_state)
        n_samples = X.shape[0]
        if self.maxneighbor is None:
            total_neighbors = self.n_clusters * (n_samples - self.n_clusters)
            self.maxneighbor = max(250, int(0.0125 * total_neighbors))
        best_cost = float('inf')
        best_medoid_indices = None
        for i in range(self.numlocal):
            current_medoid_indices = list(np.random.choice(n_samples, self.n_clusters, replace=False))
            current_medoids = X[current_medoid_indices]
            current_cost = calculate_total_cost(X, current_medoids, current_medoid_indices)
            j = 1
            while j <= self.maxneighbor:
                medoid_to_replace_idx = np.random.randint(0, self.n_clusters)
                non_medoid_indices = [idx for idx in range(n_samples) if idx not in current_medoid_indices]
                new_medoid_index = np.random.choice(non_medoid_indices)
                neighbor_medoid_indices = current_medoid_indices.copy()
                neighbor_medoid_indices[medoid_to_replace_idx] = new_medoid_index
                neighbor_medoids = X[neighbor_medoid_indices]
                neighbor_cost = calculate_total_cost(X, neighbor_medoids, neighbor_medoid_indices)
                if neighbor_cost < current_cost:
                    current_medoid_indices = neighbor_medoid_indices
                    current_medoids = neighbor_medoids
                    current_cost = neighbor_cost
                    j = 1
                    self.n_iter_ += 1
                else:
                    j += 1
            if current_cost < best_cost:
                best_cost = current_cost
                best_medoid_indices = current_medoid_indices
        self.best_medoid_indices_ = best_medoid_indices
        self.best_medoids_ = X[best_medoid_indices]
        self.cost_ = best_cost
        self.labels_ = assign_clusters(X, self.best_medoids_)
        return self
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_

# --- K-Means Implementation ---

class KMeans:
    def __init__(self, n_clusters: int, max_iter: int = 300, random_state: int = None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
    
    def fit(self, X: np.ndarray) -> 'KMeans':
        if self.random_state is not None:
            np.random.seed(self.random_state)
        n_samples = X.shape[0]
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids_ = X[random_indices].copy()
        for iteration in range(self.max_iter):
            labels = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                min_distance = float('inf')
                best_cluster = 0
                for j in range(self.n_clusters):
                    distance = euclidean_distance(X[i], self.centroids_[j])
                    if distance < min_distance:
                        min_distance = distance
                        best_cluster = j
                labels[i] = best_cluster
            new_centroids = np.zeros_like(self.centroids_)
            for j in range(self.n_clusters):
                cluster_points = X[labels == j]
                if len(cluster_points) > 0:
                    new_centroids[j] = np.mean(cluster_points, axis=0)
                else:
                    new_centroids[j] = X[np.random.choice(n_samples)]
            if np.allclose(self.centroids_, new_centroids):
                self.n_iter_ = iteration + 1
                break
            self.centroids_ = new_centroids
            self.n_iter_ = iteration + 1
        self.labels_ = labels
        self.inertia_ = 0.0
        for i in range(n_samples):
            distance = euclidean_distance(X[i], self.centroids_[self.labels_[i]])
            self.inertia_ += distance ** 2
        return self
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_

# --- Main Script ---

if __name__ == "__main__":
    try:
        # Load Data
        df = pd.read_csv('Mall_Customers.csv')
        X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
        
        # Normalize Data
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_normalized = (X - X_mean) / X_std
        
        # 1. Visualize Original Data
        plt.figure(figsize=(10, 6))
        plt.scatter(X[:, 0], X[:, 1], alpha=0.6, edgecolors='k', s=50)
        plt.xlabel('Annual Income (k$)', fontsize=12)
        plt.ylabel('Spending Score (1-100)', fontsize=12)
        plt.title('Mall Customers - Original Data', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('original_data.png')
        print("Saved original_data.png")
        
        # 2. Run CLARANS
        n_clusters = 5
        print("Fitting CLARANS...")
        clarans = CLARANS(n_clusters=n_clusters, numlocal=2, random_state=42)
        clarans_labels = clarans.fit_predict(X_normalized)
        
        # Visualize CLARANS Results
        plt.figure(figsize=(12, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i in range(n_clusters):
            cluster_points = X[clarans_labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       s=50, c=colors[i], label=f'Cluster {i}', alpha=0.6, edgecolors='k')
        medoids_original = clarans.best_medoids_ * X_std + X_mean
        plt.scatter(medoids_original[:, 0], medoids_original[:, 1], 
                   s=200, c='yellow', marker='*', edgecolor='black', label='Medoids')
        plt.title('CLARANS Clustering Results', fontsize=14, fontweight='bold')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('clarans_clusters.png')
        print("Saved clarans_clusters.png")
        
        # 3. Run K-Means
        print("Fitting K-Means...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(X_normalized)
        
        # Visualize K-Means Results
        plt.figure(figsize=(12, 6))
        for i in range(n_clusters):
            cluster_points = X[kmeans_labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       s=50, c=colors[i], label=f'Cluster {i}', alpha=0.6, edgecolors='k')
        centroids_original = kmeans.centroids_ * X_std + X_mean
        plt.scatter(centroids_original[:, 0], centroids_original[:, 1], 
                   s=200, c='yellow', marker='X', edgecolor='black', label='Centroids')
        plt.title('K-Means Clustering Results', fontsize=14, fontweight='bold')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('kmeans_clusters.png')
        print("Saved kmeans_clusters.png")
        
    except Exception as e:
        print(f"An error occurred: {e}")
