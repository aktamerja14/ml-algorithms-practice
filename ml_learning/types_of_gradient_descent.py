
import numpy as np

X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

learning_rate = 0.01
epochs = 5
n = len(X)


# Function to calculate error
def calculate_error(X, y, m, b):
    y_pred = m * X + b
    return np.mean((y - y_pred) ** 2)


# 1. Batch Gradient Descent
print("\n===== BATCH GRADIENT DESCENT =====")

m, b = 0.0, 0.0

for epoch in range(epochs):
    y_pred = m * X + b

    error = calculate_error(X, y, m, b)

    dm = (-2 / n) * np.sum(X * (y - y_pred))
    db = (-2 / n) * np.sum(y - y_pred)

    m -= learning_rate * dm
    b -= learning_rate * db

    print(
        f"Epoch {epoch+1}: "
        f"m={m:.4f}, b={b:.4f}, Error={error:.4f}"
    )


# 2. Stochastic Gradient Descent
print("\n===== STOCHASTIC GRADIENT DESCENT =====")

m, b = 0.0, 0.0

for epoch in range(epochs):
    for i in range(n):
        prediction = m * X[i] + b

        dm = -2 * X[i] * (y[i] - prediction)
        db = -2 * (y[i] - prediction)

        m -= learning_rate * dm
        b -= learning_rate * db

    error = calculate_error(X, y, m, b)

    print(
        f"Epoch {epoch+1}: "
        f"m={m:.4f}, b={b:.4f}, Error={error:.4f}"
    )


# 3. Mini-Batch Gradient Descent
print("\n===== MINI-BATCH GRADIENT DESCENT =====")

m, b = 0.0, 0.0
batch_size = 2

for epoch in range(epochs):
    for start in range(0, n, batch_size):
        X_batch = X[start:start + batch_size]
        y_batch = y[start:start + batch_size]

        y_pred = m * X_batch + b
        batch_n = len(X_batch)

        dm = (-2 / batch_n) * np.sum(
            X_batch * (y_batch - y_pred)
        )
        db = (-2 / batch_n) * np.sum(y_batch - y_pred)

        m -= learning_rate * dm
        b -= learning_rate * db

    error = calculate_error(X, y, m, b)

    print(
        f"Epoch {epoch+1}: "
        f"m={m:.4f}, b={b:.4f}, Error={error:.4f}"
    )
