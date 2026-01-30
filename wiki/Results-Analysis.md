# Results & Analysis

This section discusses the performance of CLARANS on the Mall Customers dataset and compares it with the K-Means algorithm.

## Dataset Overview

The **Mall Customers Dataset** contains customer information with features:
- **Customer ID**: Unique identifier
- **Gender**: Male/Female
- **Age**: Customer age
- **Annual Income (k$)**: Income in thousands of dollars
- **Spending Score (1-100)**: Score assigned based on customer behavior and spending patterns

For clustering analysis, we focus on **Annual Income** and **Spending Score** as these reveal distinct customer segments.

## Clustering Visualization

### Data Distribution
The dataset features distinct groups when plotting **Annual Income** vs. **Spending Score**, suggesting natural clustering patterns.

![Original Data](../Visuals/original_data.png)

*(Note: Images reference files in the repository)*

### CLARANS Results
CLARANS successfully identifies 5 major customer segments. The medoids (marked with stars ⭐) represent actual customers from the dataset, which makes the cluster centers interpretable and actionable for business decisions.

![CLARANS](../Visuals/clarans_clusters.png)

**Key observations:**
- Medoids are real data points, providing concrete customer profiles for each segment.
- Clusters show clear separation between different customer types.
- The randomized search effectively identifies optimal medoid positions.

### K-Means Results
K-Means also partitions the data effectively. Since it uses centroids (mean of points) rather than medoids, the centers (marked with X ✕) do not necessarily correspond to real data points.

![K-Means](../Visuals/kmeans_clusters.png)

**Key observations:**
- Centroids may fall in empty spaces between customers.
- Clusters are similar in structure to CLARANS results.
- Faster execution but less interpretable cluster centers.

## Quantitative Comparison

We evaluated both algorithms using **Silhouette Score** and execution time.

| Metric | CLARANS | K-Means |
|--------|---------|---------|
| **Execution Time** | Slower (Randomized Graph Search) | Faster (Iterative Refinement) |
| **Silhouette Score** | ~0.55 | ~0.55 |
| **Cluster Centers** | Medoids (Real Data Points) | Centroids (Computed Averages) |
| **Interpretability** | High - actual customers | Lower - abstract points |
| **Robustness to Outliers** | High | Low (outliers pull centroids) |

### Key Findings

1. **Quality**: On this dataset, CLARANS achieves clustering quality comparable to K-Means as measured by silhouette score.

2. **Speed**: K-Means is significantly faster because it uses a simple iterative refinement approach. CLARANS performs randomized graph search with multiple distance computations.

3. **Interpretability**: CLARANS has a clear advantage - each cluster is represented by an actual customer, making it easier to:
   - Describe customer segments to stakeholders
   - Use medoids as representative profiles for marketing
   - Make data-driven decisions based on real examples

4. **Robustness**: K-Means is sensitive to outliers as they disproportionately pull the centroid. CLARANS (using medoids) is generally more robust to outliers, though this dataset is relatively clean.

5. **Practical Application**: For customer segmentation, CLARANS provides more actionable insights despite longer runtime.

## Customer Segments Identified

Based on the visualization, both algorithms identify these approximate segments:
- **High Income, High Spending**: Premium customers
- **High Income, Low Spending**: Target for upselling
- **Medium Income, Medium Spending**: Average customers
- **Low Income, High Spending**: Budget shoppers with high engagement
- **Low Income, Low Spending**: Minimal engagement group

## Conclusion

CLARANS proves to be a viable and often superior alternative to K-Means, particularly for applications where:
- The "center" of a cluster must be an actual object (e.g., selecting a representative customer profile).
- Interpretability of results is crucial for business decisions.
- Robustness to outliers is important.
- The dataset is large enough that PAM becomes infeasible, but quality remains critical.

For the Mall Customers dataset, while both algorithms produce similar clustering structures, CLARANS offers the additional benefit of providing real customer profiles as cluster representatives, making the results more actionable for marketing and business strategy.
