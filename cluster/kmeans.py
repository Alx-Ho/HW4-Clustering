import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        # Basic argument validation to guard against incorrect usage.
        if not isinstance(k, int):
            raise TypeError("k must be an integer")
        if k <= 0:
            raise ValueError("k must be greater than 0")
        if not isinstance(tol, (int, float)) or tol <= 0:
            raise ValueError("tol must be a positive number")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")

        # Store hyperparameters and initialize internal state.
        self.k = k
        self.tol = float(tol)
        self.max_iter = max_iter
        self._centroids = None
        self._error = None
        self._n_features = None
        self._fitted = False

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        # Validate the input matrix shape and size.
        if not isinstance(mat, np.ndarray):
            raise TypeError("mat must be a numpy array")
        if mat.ndim != 2:
            raise ValueError("mat must be a 2D matrix")
        if mat.shape[0] < self.k:
            raise ValueError("number of observations must be >= k")

        n_samples, n_features = mat.shape
        self._n_features = n_features

        # Initialize centroids by sampling existing points.
        rng = np.random.default_rng(42)
        init_idx = rng.choice(n_samples, size=self.k, replace=False)
        centroids = mat[init_idx].astype(float)

        prev_error = None
        for _ in range(self.max_iter):
            # Assign each point to its nearest centroid.
            distances = cdist(mat, centroids)
            labels = np.argmin(distances, axis=1)

            # Update centroids based on current assignments.
            new_centroids = np.zeros_like(centroids)
            for idx in range(self.k):
                members = mat[labels == idx]
                if members.size == 0:
                    # If a centroid loses all members, re-seed it to a random point.
                    new_centroids[idx] = mat[rng.integers(0, n_samples)]
                else:
                    new_centroids[idx] = np.mean(members, axis=0)

            # Compute mean squared distance to nearest centroid as error.
            distances = cdist(mat, new_centroids)
            closest = np.min(distances, axis=1)
            error = float(np.mean(closest ** 2))

            # Stop if improvement is below the tolerance threshold.
            if prev_error is not None and abs(prev_error - error) < self.tol:
                centroids = new_centroids
                prev_error = error
                break

            centroids = new_centroids
            prev_error = error

        # Persist learned parameters.
        self._centroids = centroids
        self._error = prev_error
        self._fitted = True

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        # Ensure the model is trained and inputs are compatible.
        if not self._fitted:
            raise RuntimeError("model must be fit before prediction")
        if not isinstance(mat, np.ndarray):
            raise TypeError("mat must be a numpy array")
        if mat.ndim != 2:
            raise ValueError("mat must be a 2D matrix")
        if mat.shape[1] != self._n_features:
            raise ValueError("mat must have the same number of features as the fit data")

        # Assign each observation to the nearest centroid.
        distances = cdist(mat, self._centroids)
        return np.argmin(distances, axis=1)

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        # Error is only available after fitting.
        if not self._fitted:
            raise RuntimeError("model must be fit before getting error")
        return float(self._error)

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        # Centroids are only available after fitting.
        if not self._fitted:
            raise RuntimeError("model must be fit before getting centroids")
        return self._centroids.copy()
