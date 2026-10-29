
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. Create sample data
np.random.seed(42)

X = np.random.randn(100, 3)

# X1 and X2 are useful; X3 is unnecessary
y = 5 * X[:, 0] + 2 * X[:, 1]
y += np.random.normal(0, 0.5, 100)

# 2. Train Linear Regression
linear_model = make_pipeline(
    StandardScaler(),
    LinearRegression()
)
linear_model.fit(X, y)

# 3. Train Lasso Regression
lasso_model = make_pipeline(
    StandardScaler(),
    Lasso(alpha=0.1, max_iter=10000)
)
lasso_model.fit(X, y)

# 4. Get coefficients
linear_coefficients = linear_model.named_steps[
    "linearregression"
].coef_

lasso_coefficients = lasso_model.named_steps[
    "lasso"
].coef_

print("Linear Regression Coefficients:")
print(np.round(linear_coefficients, 4))

print("\nLasso Regression Coefficients:")
print(np.round(lasso_coefficients, 4))

# 5. Plot comparison
features = ["X1", "X2", "X3"]
positions = np.arange(len(features))
width = 0.35

plt.bar(
    positions - width / 2,
    linear_coefficients,
    width,
    label="Linear Regression"
)

plt.bar(
    positions + width / 2,
    lasso_coefficients,
    width,
    label="Lasso Regression"
)

plt.axhline(0, color="black", linewidth=0.8)
plt.xticks(positions, features)
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.title("Linear Regression vs Lasso")
plt.legend()
plt.grid(axis="y")
plt.show()
