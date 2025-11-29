# Testing Linear regression with 

import numpy as np
from linear_regression import LinearRegression

np.random.seed(42)

X = np.random.rand(100, 1)
y = 4 * X + 2 + np.random.randn(100, 1) * 0.1
y = y.ravel()   

model = LinearRegression(learning_rate=0.05, n_iterations=2000)
model.fit(X, y)


print("Weights:", model.weights)
print("Bias:", model.bias)

print("First 5 predictions:", model.predict(X[:5]))
