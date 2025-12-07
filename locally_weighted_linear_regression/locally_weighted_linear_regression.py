import numpy as np

class locally_weighted_linear_regression:
    def __init__(self, tau = 1.0):
        self.tau = tau
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y

    def get_weights(self, x_query):

        # Compute squared distances for each training point
        diff = self.X - x_query
        distances = np.sum(diff ** 2, axis=1)

        # Gaussian kernel
        weights = np.exp(-distances / (2*self.tau ** 2))
        
        # Return diagonal matrix
        return np.diag(weights)
    
    def predict_one(self, x_query):
        W = self.get_weights(x_query)

        # Normal equation with weights
        XTWX = self.X.T @ W @ self.X
        XTWy = self.X.T @ W @ self.y

        # Add small identity matrix to avoid singularity
        try:
            theta = np.linalg.solve(XTWX, XTWy)
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(XTWX) @ XTWy

        return x_query @ theta
    
    def predict(self, X_query):
        predictions = [self.predict_one(x) for x in X_query]
        return np.array(predictions)

