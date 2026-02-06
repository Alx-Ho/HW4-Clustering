import numpy as np
import pytest
from sklearn.metrics import silhouette_samples

from cluster import KMeans, Silhouette, make_clusters


def test_silhouette_matches_sklearn_mean():
	# Build clustered data and fit KMeans to obtain labels.
	mat, _ = make_clusters(n=150, m=2, k=3, scale=0.6, seed=3)
	model = KMeans(k=3, tol=1e-5, max_iter=100)
	model.fit(mat)
	labels = model.predict(mat)

	# Compare mean silhouette score with sklearn implementation.
	ours = Silhouette().score(mat, labels)
	skl = silhouette_samples(mat, labels, metric="euclidean")

	assert ours.shape == skl.shape
	assert np.allclose(np.mean(ours), np.mean(skl), atol=1e-2)


def test_silhouette_rejects_single_cluster():
	# Single-cluster labeling should be rejected.
	mat = np.random.default_rng(0).normal(size=(10, 2))
	labels = np.zeros(10, dtype=int)

	with pytest.raises(ValueError):
		Silhouette().score(mat, labels)
