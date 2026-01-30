# Algorithm Theory

## Concepts

**CLARANS** (Clustering Large Applications based on RANdomized Search) was proposed by Ng and Han in 1994 to address the scalability issues of partitioning methods like PAM (Partitioning Around Medoids).

### 1. The Graph Model
CLARANS views the clustering process as a search through a graph where:
- **Nodes**: Each node represents a set of \(k\) medoids.
- **Neighbors**: Two nodes are neighbors if their sets of medoids differ by exactly one object.

### 2. Philosophical Approach
- **PAM**: Searches the *entire* graph exhaustively. This is optimal but computationally expensive (\(O(k(n-k)^2)\) per iteration).
- **CLARA**: Takes a small sample of the data and runs PAM on it. This is fast but may miss optimal solutions if the sample is not representative.
- **CLARANS**: Draws a middle ground. It searches the original graph but does not check *every* neighbor of the current node. Instead, it checks a random sample of neighbors.

## Key Parameters

CLARANS is governed by two main parameters:

### `numlocal`
- **Definition**: The number of local minima to find.
- **Role**: Since CLARANS uses a randomized greedy search, it may get stuck in a local minimum. To increase the probability of finding a good global solution, the algorithm is run multiple times (`numlocal` times), and the best result (lowest cost) is returned.
- **Typical Value**: 2 (recommended by the original paper).

### `maxneighbor`
- **Definition**: The maximum number of neighbors to examine for any current node.
- **Role**: Limits the search scope. If the algorithm checks `maxneighbor` random neighbors and doesn't find one with a lower cost, the current node is declared a local minimum.
- **Typical Value**: The paper suggests determining this based on the dataset size, often \(1.25\%\) of \(k(n-k)\).

## The CLARANS Algorithm

1. Initialize `i = 1` and `minCost = infinity`.
2. **Main Loop**: Repeat `numlocal` times:
    a. Select a random node \(S\) (set of \(k\) medoids).
    b. Set `j = 1`.
    c. **Neighbor Search**: Repeat while `j <= maxneighbor`:
        i. Pick a random neighbor \(S'\) of \(S\).
        ii. Calculate cost difference.
        iii. **Improvement?**
            - **Yes**: Set \(S = S'\), reset `j = 1`, and continue.
            - **No**: Increment `j`.
    d. **Local Minimum Found**: When `j > maxneighbor`, \(S\) is a local minimum.
    e. If `Cost(S) < minCost`, update `minCost` and store best medoids.
3. Return the best set of medoids found.

## Complexity Analysis

- **PAM**: \(O(k(n-k)^2)\) per iteration - examines all neighbors exhaustively.
- **CLARA**: \(O(k(40+k)^2)\) per iteration on sample + \(O(kn)\) for assignment - fast but quality depends on sample representativeness.
- **CLARANS**: Approximately linear in \(n\) - randomized search provides efficiency without sacrificing quality.

## Advantages of CLARANS

1. **Scalability**: More efficient than PAM for large datasets.
2. **Quality**: Produces higher quality clusters than CLARA by searching the full graph space.
3. **Flexibility**: Works with any distance metric, not just Euclidean.
4. **Robustness**: Medoid-based approach is robust to outliers.
