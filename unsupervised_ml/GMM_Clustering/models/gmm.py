from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


class GMMClustering:

    def __init__(self, n_components=3):
        self.n_components = n_components

    def fit(self, X):

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        model = GaussianMixture(
            n_components=self.n_components,
            random_state=42
        )

        labels = model.fit_predict(X_scaled)

        probabilities = model.predict_proba(X_scaled)

        score = silhouette_score(
            X_scaled,
            labels
        )

        return {
            "X_scaled": X_scaled,
            "labels": labels,
            "score": score,
            "probabilities": probabilities,
            "means": model.means_
        }