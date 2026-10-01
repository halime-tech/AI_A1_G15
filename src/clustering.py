from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def evaluate_k(X, k_range=range(2, 6), seed=42):
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    results = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(Xs)
        results[k] = {
            "silhouette": float(silhouette_score(Xs, labels)),
            "inertia": float(km.inertia_),
        }
    return scaler, results


def final_cluster(X, k, seed=42):
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(Xs)
    return scaler, km, labels