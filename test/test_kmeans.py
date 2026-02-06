import numpy as np
import pytest

from cluster import KMeans, make_clusters


def test_kmeans_fit_predict_shapes():
	# Generate a small synthetic dataset with known dimensions.
	mat, _ = make_clusters(n=200, m=3, k=4, scale=0.5, seed=7)
	# Fit model and collect outputs.
	model = KMeans(k=4, tol=1e-5, max_iter=100)
	model.fit(mat)

	labels = model.predict(mat)
	centroids = model.get_centroids()
	error = model.get_error()

	# Validate output shapes and reasonable values.
	assert labels.shape == (mat.shape[0],)
	assert centroids.shape == (4, mat.shape[1])
	assert np.unique(labels).size <= 4
	assert error > 0


def test_kmeans_predict_feature_mismatch():
	# Fit on a 2D feature matrix.
	mat, _ = make_clusters(n=50, m=2, k=3, scale=0.8, seed=11)
	model = KMeans(k=3)
	model.fit(mat)

	# Predict on mismatched feature dimensions should raise.
	bad_mat = np.random.default_rng(0).normal(size=(10, 3))
	with pytest.raises(ValueError):
		model.predict(bad_mat)


def test_kmeans_rejects_zero_k():
	# k=0 should be rejected at initialization.
	with pytest.raises(ValueError):
		KMeans(k=0)


def test_kmeans_rejects_more_clusters_than_points():
	# Fitting with fewer observations than k should raise.
	mat = np.random.default_rng(1).normal(size=(3, 2))
	model = KMeans(k=4)
	with pytest.raises(ValueError):
		model.fit(mat)


def test_kmeans_handles_high_k():
	# High k (close to n) should still run.
	mat = np.random.default_rng(2).normal(size=(20, 2))
	model = KMeans(k=18, tol=1e-5, max_iter=50)
	model.fit(mat)
	labels = model.predict(mat)
	assert labels.shape == (20,)


def test_kmeans_handles_high_dimensionality():
	# High dimensional input should be accepted.
	mat, _ = make_clusters(n=120, m=60, k=4, scale=1.0, seed=5)
	model = KMeans(k=4, tol=1e-5, max_iter=100)
	model.fit(mat)
	labels = model.predict(mat)
	assert labels.shape == (120,)


def test_kmeans_handles_single_dimension():
	# Single-feature input should be accepted.
	mat, _ = make_clusters(n=100, m=1, k=3, scale=0.7, seed=9)
	model = KMeans(k=3, tol=1e-5, max_iter=100)
	model.fit(mat)
	labels = model.predict(mat)
	assert labels.shape == (100,)
