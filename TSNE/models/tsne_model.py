from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


class TSNEModel:

    def __init__(
        self,
        perplexity=30,
        n_components=2
    ):
        self.perplexity = perplexity
        self.n_components = n_components

    def fit(self, X):

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        tsne = TSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            random_state=42
        )

        embedding = tsne.fit_transform(X_scaled)

        return {
            "embedding": embedding
        }