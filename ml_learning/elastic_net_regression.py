
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 1. Create sample data
np.random.seed(42)

X = np.random.randn(100, 4)

# First two features are useful
# Last two features are unnecessary
y = 5 * X[:, 0] + 2 * X[:, 1]
y += np.random.normal(0, 0.5, 100)

# 2. Create models
models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1, max_iter=10000),
    "Elastic Net": ElasticNet(
        alpha=0.1,
        l1_ratio=0.5,
        max_iter=10000
    )
}

coefficients = {}
errors = {}

# 3. Train each model
for name, model in models.items():

    pipeline = make_pipeline(
        StandardScaler(),
        model
    )

    pipeline.fit(X, y)

    y_pred = pipeline.predict(X)

    coef = pipeline.named_steps[
        model.__class__.__name__.lower()
    ].coef_

    coefficients[name] = coef
    errors[name] = mean_squared_error(y, y_pred)

    print(f"\n{name} Regression")
    print("Coefficients:", np.round(coef, 3))
    print("MSE:", round(errors[name], 3))

# 4. Compare coefficients
features = ["X1", "X2", "X3", "X4"]
positions = np.arange(len(features))
width = 0.2

plt.figure(figsize=(10, 5))

for i, (name, coef) in enumerate(coefficients.items()):
    plt.bar(
        positions + (i - 1.5) * width,
        coef,
        width,
        label=name
    )

plt.axhline(0, color="black", linewidth=0.8)
plt.xticks(positions, features)
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.title("Linear vs Ridge vs Lasso vs Elastic Net")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.show()
