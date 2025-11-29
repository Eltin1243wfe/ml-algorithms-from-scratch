# Linear Regression - Machine Learning Algorithim from scratch

This module implements Linear Regression from scratch only using NumPy library
The goal is to understand how gradient descent, weight and bias updates, cost function and linear algebra works internally without scikit-learn

The project includes:
--Linear regression class
--Manual testing
--Synthetic testing
--Using .csv file testing

# Directory structure:
linear_regression/
│── linear_regression.py        # Algorithm implementation
│── test_manual.py              # Simple manual test
│── test_synthetic.py           # Random synthetic dataset test
│── test_csv.py                 # Real CSV dataset test
│── linear_data.csv             # Data file for CSV tests

# Math explanation
Hypothesis:  y(x) = w * x + b

Cost function: L = (1/2m) * np.sum(pow((y(x) - y)),2)

Gradients: 
    dw = (1/m) * pow(X,T) * (y(x) - y)
    db = (1/m) * sum(y(x) - y)

Updating weights and bias:
    weights = weight - α * dw
    bias = bias - α * db

    α -> learning rate

# 🙌 Notes

This is my first README.md file.
My learning sources include:
    CS229 by Andrew Ng
    Additional documentation and ML tutorials online

If you have suggestions or improvements, feel free to share —
I'm always open to learning and discussing ideas.

Thank you!
