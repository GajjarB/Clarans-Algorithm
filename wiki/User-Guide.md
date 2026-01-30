# User Guide

This guide helps you set up the environment and run the CLARANS clustering analysis.

## Prerequisites

Ensure you have Python 3.8+ installed. You will need the following dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn
```

**Note**: `scikit-learn` is included for comparison with K-Means and computing evaluation metrics like Silhouette Score.

## Directory Structure

```text
.
├── Mall_Customers.csv          # Input dataset
├── clarans_implementation.ipynb # Exploratory Jupyter Notebook
├── generate_plots.py           # Script to run algorithms and save plots
├── README.md                   # Main documentation
├── Visuals/                    # Output folder for images
│   ├── original_data.png
│   ├── clarans_clusters.png
│   └── kmeans_clusters.png
└── wiki/                       # Documentation files
    ├── Algorithm-Theory.md
    ├── Implementation.md
    ├── User-Guide.md
    ├── Results-Analysis.md
    └── Home.md
```

## Running the Project

### Option 1: Jupyter Notebook (Recommended for Learning)
For an interactive experience where you can step through the logic:
1. Launch Jupyter Lab or Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `clarans_implementation.ipynb`.
3. Run all cells to see the step-by-step execution:
   - Data loading and exploration
   - CLARANS implementation
   - K-Means comparison
   - Visualization and analysis

### Option 2: Python Script (Quick Results)
To quickly generate results and visualizations:
1. Open your terminal in the project directory.
2. Run the generation script:
   ```bash
   python generate_plots.py
   ```
3. Check the `Visuals/` folder (or root directory) for the generated PNG files.

## Customization

You can tweak the algorithm parameters in `generate_plots.py` or the notebook cells:

### Adjusting Number of Clusters
```python
# Change number of clusters
n_clusters = 5  # Try different values: 3, 4, 5, 6, etc.
```

### Tuning CLARANS Parameters
```python
# Adjust CLARANS search parameters
clarans = CLARANS(
    n_clusters=5,
    numlocal=2,        # Number of local minima to find
    maxneighbor=250,   # Maximum neighbors to examine
    random_state=42    # For reproducibility
)
```

**Parameter Guidelines:**
- **`numlocal`**: 
  - Default: 2 (recommended)
  - Higher values: Better chance of finding global optimum, but slower
  - Lower values: Faster but may miss optimal solution

- **`maxneighbor`**: 
  - Default: 250-500 for this dataset size
  - Higher values: More thorough search, better quality, slower
  - Lower values: Faster but potentially lower quality
  - Recommended: 1.25% of k(n-k) where k=clusters, n=data points

- **`random_state`**: 
  - Set to a fixed value (e.g., 42) for reproducible results
  - Set to `None` for different results each run

### Changing Dataset Features
```python
# Select different features for clustering
X = data[['Age', 'Annual Income (k$)']].values  # Different feature combination
```

## Understanding the Output

### Console Output
The script will print:
- Number of data points loaded
- Clustering parameters used
- Execution time for each algorithm
- Silhouette scores (quality metric)

### Visualizations
Three plots are generated:
1. **Original Data**: Shows the distribution of data points
2. **CLARANS Clusters**: Shows clusters with medoids marked as stars
3. **K-Means Clusters**: Shows clusters with centroids marked as X

### Interpreting Results
- **Silhouette Score**: Range [-1, 1]
  - Closer to 1: Well-separated clusters
  - Around 0: Overlapping clusters
  - Negative: Points may be in wrong clusters
- **Visual Inspection**: Check if clusters make intuitive sense for your domain

## Troubleshooting

### Import Errors
If you encounter import errors:
```bash
pip install --upgrade numpy pandas matplotlib scikit-learn
```

### Memory Issues
For very large datasets:
- Reduce `maxneighbor` value
- Process data in batches
- Use dimensionality reduction first

### Poor Clustering Quality
If results are unsatisfactory:
- Increase `maxneighbor` for more thorough search
- Increase `numlocal` to find better local minima
- Try different numbers of clusters (k)
- Normalize/standardize your features
- Check for outliers in your data

## Performance Tips

1. **Start small**: Test with `maxneighbor=250` and `numlocal=2`
2. **Scale up gradually**: Increase parameters if quality is insufficient
3. **Use random_state**: Set it during development for consistent results
4. **Profile your code**: Use timing functions to identify bottlenecks
5. **Vectorize operations**: The implementation uses NumPy for efficiency
