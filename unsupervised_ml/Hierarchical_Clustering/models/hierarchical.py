from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

class HierarchicalClustering:

    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters

    def fit(self, X):

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        model = AgglomerativeClustering(
            n_clusters=self.n_clusters
        )

        labels = model.fit_predict(X_scaled)

        score = silhouette_score(
            X_scaled,
            labels
        )

        return {
            "X_scaled": X_scaled,
            "labels": labels,
            "score": score
        }