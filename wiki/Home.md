# Welcome to the CLARANS Project Wiki

This wiki documents the implementation and usage of the **CLARANS** (Clustering Large Applications based on RANdomized Search) algorithm for customer segmentation.

## 📖 Navigation

1. **[Algorithm Theory](Algorithm-Theory.md)**: Deep dive into how CLARANS works and why it is efficient for large datasets.
2. **[Implementation Details](Implementation.md)**: Overview of the Python codebase, key classes, and functions.
3. **[User Guide](User-Guide.md)**: Step-by-step instructions on running the project and using the scripts.
4. **[Results & Analysis](Results-Analysis.md)**: Interpretation of the clustering results and performance metrics compared to K-Means.

## 🚀 Quick Summary

**CLARANS** (proposed by Ng and Han, 1994) strikes a balance between the effectiveness of PAM (Partitioning Around Medoids) and the efficiency of CLARA. It views the process of finding \(k\) medoids as searching through a graph. By conducting a randomized search, it avoids the exhaustive computation of checking every neighbor (like PAM) while maintaining better quality than random sampling (like CLARA).

### Key Features

- **Graph-based approach**: Views clustering as searching through a graph where nodes represent medoid sets.
- **Randomized search**: Examines only a sample of neighbors at each step, balancing efficiency and quality.
- **Medoid-based**: Uses actual data points as cluster centers, making results interpretable and robust to outliers.
- **Scalable**: Designed specifically for large datasets where PAM becomes computationally prohibitive.

## Project Overview

This project demonstrates:
- A custom Python implementation of CLARANS from scratch.
- Application on the **Mall Customers Dataset** for customer segmentation.
- Comparative analysis against standard K-Means clustering.
- Visualization of clustering results showing distinct customer segments.
