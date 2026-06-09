from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


class DBSCANClustering:

    def __init__(
        self,
        eps=0.5,
        min_samples=5
    ):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples
        )

        labels = model.fit_predict(X_scaled)

        return {
            "X_scaled": X_scaled,
            "labels": labels
        }