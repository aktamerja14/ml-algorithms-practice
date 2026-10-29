
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

# 1. Sample data
np.random.seed(42)

X = np.linspace(0, 10, 30).reshape(-1, 1)
y = 2 * X.ravel() + np.sin(2 * X.ravel()) * 3
y += np.random.normal(0, 2, size=len(X))

# 2. Create models
# Degree 10 model can become too complex
normal_model = make_pipeline(
    PolynomialFeatures(degree=10),
    StandardScaler(),
    LinearRegression()
)

regularized_model = make_pipeline(
    PolynomialFeatures(degree=10),
    StandardScaler(),
    Ridge(alpha=10)
)

# 3. Train models
normal_model.fit(X, y)
regularized_model.fit(X, y)

# 4. Predictions
X_curve = np.linspace(0, 10, 200).reshape(-1, 1)

y_normal = normal_model.predict(X_curve)
y_regularized = regularized_model.predict(X_curve)

print("Normal Regression MSE:",
      round(mean_squared_error(y, normal_model.predict(X)), 2))

print("Regularized Regression MSE:",
      round(mean_squared_error(y, regularized_model.predict(X)), 2))

# 5. Plot graph
plt.scatter(X, y, label="Actual Data")
plt.plot(X_curve, y_normal, label="Normal Regression")
plt.plot(X_curve, y_regularized, label="Regularized Regression")

plt.xlabel("X")
plt.ylabel("y")
plt.title("Regularization: Model Comparison")
plt.legend()
plt.grid(True)
plt.show()
