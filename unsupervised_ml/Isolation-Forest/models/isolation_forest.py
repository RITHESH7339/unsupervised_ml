from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
class IsolationForestModel:

    def __init__(
        self,
        contamination=0.05
    ):
        self.contamination = contamination

    def fit(self, X):

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(
            contamination=self.contamination,
            random_state=42
        )

        predictions = model.fit_predict(X_scaled)

        anomaly_scores = model.decision_function(
            X_scaled
        )

        return {
            "X_scaled": X_scaled,
            "predictions": predictions,
            "scores": anomaly_scores
        }