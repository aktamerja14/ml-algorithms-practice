
import numpy as np

X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

m = 0.0
b = 0.0

learning_rate = 0.01
epochs = 10
n = len(X)

for i in range(epochs):

    # 1. Prediction
    y_pred = m * X + b

    # 2. Error
    error = np.mean((y - y_pred) ** 2)

    # 3. Gradient calculation
    dm = (-2 / n) * np.sum(X * (y - y_pred))
    db = (-2 / n) * np.sum(y - y_pred)

    # 4. Update m and b
    m = m - learning_rate * dm
    b = b - learning_rate * db

    print(
        f"Epoch {i+1}: "
        f"m={m:.4f}, b={b:.4f}, "
        f"Error={error:.4f}"
    )
