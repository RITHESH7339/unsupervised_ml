from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCAModel:

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        pca = PCA(
            n_components=self.n_components
        )

        components = pca.fit_transform(X_scaled)

        explained_variance = (
            pca.explained_variance_ratio_
        )

        return {
            "X_scaled": X_scaled,
            "components": components,
            "variance": explained_variance
        }