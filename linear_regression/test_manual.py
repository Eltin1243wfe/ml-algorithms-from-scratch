# Testing Linear regression with simple dataset

import numpy as np
from linear_regression import LinearRegression

X = np.array([[1], [2], [3], [4]], dtype=float)
y = np.array([3, 5, 7, 9], dtype=float)

model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)

print("Weights:", model.weights)
print("Bias:", model.bias)

print("Predictions:", model.predict(X))
