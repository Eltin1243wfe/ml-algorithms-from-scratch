# test_csv.py
import numpy as np
from linear_regression import LinearRegression
import csv

def load_csv(path):
    X = []
    y = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append([float(row["x"])])
            y.append(float(row["y"]))
    return np.array(X), np.array(y)

def test_csv():
    X, y = load_csv("linear_data.csv")

    model = LinearRegression(learning_rate=0.01, n_iterations=1500)
    model.fit(X, y)

    preds = model.predict(X[:5])
    print("CSV Test")
    print("Weights:", model.weights)
    print("Bias:", model.bias)
    print("First 5 predictions:", preds)

if __name__ == "__main__":
    test_csv()
