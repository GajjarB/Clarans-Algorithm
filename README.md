# CLARANS: Clustering Large Applications based on RANdomized Search

This repository contains an implementation of the **CLARANS** (Clustering Large Applications based on RANdomized Search) algorithm, applied to customer segmentation using the Mall Customers dataset.

## Project Overview

CLARANS is an efficient partitioning-based clustering algorithm designed for large spatial data mining. It improves upon traditional algorithms like PAM (Partitioning Around Medoids) and CLARA by using a randomized search approach to explore the "graph" of possible medoid sets.

### Key Features
- **Randomized Search**: Efficiently explores the search space by randomly sampling neighbors.
- **Custom Implementation**: Built from scratch using Python and NumPy.
- **Comparison**: Includes a baseline comparison with the k-Means algorithm.
- **Evaluation Metrics**: Uses Silhouette Score and Davies-Bouldin Index to assess clustering quality.
- **Visualization**: Detailed plots showing original data distributions and final clustering results.

## Visualizations

````carousel
![Original Data](/AI&BD/original_data.png)
<!-- slide -->
![CLARANS Clustering](/AI&BD/clarans_clusters.png)
<!-- slide -->
![K-Means Clustering](/AI&BD/kmeans_clusters.png)
````

## Dataset

The project uses the **Mall Customers Dataset**, which is a popular dataset for customer segmentation.
- **Features used**: Annual Income (k$) and Spending Score (1-100).
- **Preprocessing**: Data is normalized to ensures equal weighting of features during distance calculations.

## Getting Started

### Prerequisites
To run the analysis, you will need the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`

### Usage
1. Ensure `Mall_Customers.csv` is in the same directory as the notebook.
2. Open and run the `clarans_implementation.ipynb` Jupyter notebook.

## Algorithm Details

CLARANS treats the clustering problem as searching through a graph where each node represents a set of $k$ medoids. It uses two key parameters:
- `numlocal`: The number of local minima to search for.
- `maxneighbor`: The maximum number of neighbors examined for each local minimum.

By randomly jumping between nodes (sets of medoids) only when a better configuration is found, CLARANS achieves a balance between the thoroughness of PAM and the efficiency of CLARA.

## References

- Ng, R. T., & Han, J. (2002). **CLARANS: A Method for Clustering Objects for Spatial Data Mining**. IEEE Transactions on Knowledge and Data Engineering.
