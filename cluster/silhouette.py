import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """
        # Placeholder state flag for symmetry with other estimator-style APIs.
        self._fitted = False

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        # Validate input shapes and alignment.
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if X.ndim != 2:
            raise ValueError("X must be a 2D matrix")
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of observations")

        n_samples = X.shape[0]
        if n_samples < 2:
            raise ValueError("at least 2 observations are required")

        unique_labels = np.unique(y)
        if unique_labels.size < 2:
            raise ValueError("at least 2 clusters are required for silhouette scoring")

        # Precompute all pairwise distances once for efficiency.
        distances = cdist(X, X)
        scores = np.zeros(n_samples, dtype=float)

        for idx in range(n_samples):
            label = y[idx]
            same_mask = y == label
            other_mask = y != label

            same_mask[idx] = False
            if np.sum(same_mask) == 0:
                # Silhouette is undefined for singletons; return 0 by convention.
                scores[idx] = 0.0
                continue

            # Mean intra-cluster distance (a).
            a = np.mean(distances[idx, same_mask])

            # Mean nearest-cluster distance (b).
            b = np.inf
            for other_label in unique_labels:
                if other_label == label:
                    continue
                cluster_mask = y == other_label
                if np.any(cluster_mask):
                    b = min(b, float(np.mean(distances[idx, cluster_mask])))

            # Silhouette score for the observation.
            scores[idx] = (b - a) / max(a, b)

        # Mark as computed (useful if extended later).
        self._fitted = True
        return scores
