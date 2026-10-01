import numpy as np


def train_test_split_np(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_size)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize(X_train, X_test):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma


def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def gradient_descent(X, y, lr=0.05, epochs=2000):
    Xb = add_bias(X)
    n, d = Xb.shape
    w = np.zeros(d)
    losses = []
    for _ in range(epochs):
        pred = Xb @ w
        err = pred - y
        grad = (2 / n) * (Xb.T @ err)
        w -= lr * grad
        losses.append(float(np.mean(err ** 2)))
    return w, losses


def predict(X, w):
    return add_bias(X) @ w


def metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}