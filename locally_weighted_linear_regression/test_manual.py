import numpy as np
from locally_weighted_linear_regression import locally_weighted_linear_regression

# Example data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 9])

# Fit LWLR model
model = locally_weighted_linear_regression(tau=1.0)
model.fit(X, y)

# Predict
X_test = np.array([[2.5], [3.7]])
preds = model.predict(X_test)

print(preds)
