# Implementation Details

The project is implemented in Python using `numpy` for efficient numerical computations. The core logic is encapsulated within the `CLARANS` class.

## Core Components

### 1. Distance Metric
We use **Euclidean Distance** to measure the dissimilarity between data points:
```python
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))
```

### 2. Cost Calculation
The objective function is the sum of dissimilarities between every object and its nearest medoid:
```python
def calculate_total_cost(data, medoids, medoid_indices):
    # Sum of distances from each point to its closest medoid
    total_cost = 0
    for i, point in enumerate(data):
        min_dist = float('inf')
        for medoid_idx in medoid_indices:
            dist = euclidean_distance(point, data[medoid_idx])
            min_dist = min(min_dist, dist)
        total_cost += min_dist
    return total_cost
```

### 3. The `CLARANS` Class

#### `__init__(self, n_clusters, numlocal, maxneighbor, random_state)`
Initializes the algorithm configuration:
- `n_clusters`: The target number of clusters (\(k\)).
- `numlocal`: Number of local minima to find (default: 2).
- `maxneighbor`: Maximum random neighbors to check before declaring a local optimum.
- `random_state`: Seed for reproducibility.

#### `fit(self, X)`
The main driver of the algorithm:
1. Validates and normalizes parameters (e.g., setting default `maxneighbor` if not provided).
2. Implements the nested loops described in the [Algorithm Theory](Algorithm-Theory.md).
3. For each of `numlocal` iterations:
   - Starts with random medoids.
   - Performs randomized search through the graph of possible medoid configurations.
   - Tracks the best solution found.
4. Stores `best_medoids_` and `labels_` upon completion.

#### `predict(self, X)`
Assigns new data points to the nearest medoid from the fitted model.

### 4. Neighbor Generation
A key operation is generating a random neighbor of the current medoid configuration:
- Randomly select one current medoid to replace.
- Randomly select one non-medoid point as the replacement.
- This creates a new configuration that differs by exactly one medoid.

## Helper Script: `generate_plots.py`

This script serves as an automation tool to:
1. Load the raw `Mall_Customers.csv` data.
2. Preprocess the data (select relevant features, normalize if needed).
3. Train the **CLARANS** model with optimized parameters.
4. Train a baseline **K-Means** model for comparison.
5. Generate visualization plots showing:
   - Original data distribution
   - CLARANS clustering results with medoids marked
   - K-Means clustering results with centroids marked
6. Save visualizations as image files for documentation.

## Requirements
The implementation relies on standard data science libraries:
- `numpy`: Matrix operations and numerical computations.
- `pandas`: Data loading and manipulation.
- `matplotlib`: Visualization and plotting.

## Performance Considerations

- **Memoization**: For efficiency, distances between points can be cached to avoid recomputation.
- **Parameter tuning**: The `maxneighbor` parameter significantly affects the trade-off between runtime and solution quality.
- **Random state**: Setting a random state ensures reproducible results for debugging and comparison.
