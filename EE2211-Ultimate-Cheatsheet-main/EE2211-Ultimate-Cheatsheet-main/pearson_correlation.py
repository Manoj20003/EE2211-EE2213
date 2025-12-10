import numpy as np


def pearson_correlation(X, Y):
    """Compute Pearson correlations between each feature column in X and target vector Y.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    Y : np.ndarray, shape (n_samples,) or any shape that can be raveled to (n_samples,)
        Target values.

    Returns
    -------
    corr : np.ndarray, shape (n_features,)
        Pearson correlation coefficient for each feature with Y.
    """
    X = np.asarray(X)
    Y = np.ravel(np.asarray(Y))  # flatten in case passed as (1, n)

    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional (n_samples, n_features)")
    n_samples, n_features = X.shape
    if Y.shape[0] != n_samples:
        raise ValueError("X and Y must have the same number of samples")

    # Center data
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean()
    X_centered = X - X_mean
    Y_centered = Y - Y_mean

    # Standard deviations (population std to match original intent)
    X_std = X_centered.std(axis=0)
    Y_std = Y_centered.std()

    # Covariance numerator for each feature: sum_i (x_i - mean_x) * (y_i - mean_y)
    cov_num = (X_centered * Y_centered[:, None]).sum(axis=0)
    denom = n_samples * X_std * Y_std

    corr = np.zeros(n_features, dtype=float)
    valid = denom != 0
    corr[valid] = cov_num[valid] / denom[valid]
    # Features or target with zero variance get correlation 0 (could also be NaN)
    return corr


if __name__ == "__main__":
    X = np.array([[45, 26, 3],
                  [68, 19, 7],
                  [32, 25, 5],
                  [55, 35, 9],
                  [28, 21, 2]])
    Y = np.array([2, 5, 3, 7, 2])
    corr = pearson_correlation(X, Y)
    print("Pearson correlations (feature columns vs Y):", corr)