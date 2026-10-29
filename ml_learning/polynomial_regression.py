
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 1. Sample data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([2, 5, 10, 17, 26, 37, 50, 65, 82, 101])

# 2. Simple Linear Regression
linear_model = LinearRegression()
linear_model.fit(X, y)

# 3. Polynomial Regression (degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# 4. Predictions
y_linear = linear_model.predict(X)

y_poly = poly_model.predict(X_poly)

print("Linear Regression R2:", linear_model.score(X, y))
print("Polynomial Regression R2:", poly_model.score(X_poly, y))

print("Polynomial coefficients:", poly_model.coef_)
print("Polynomial intercept:", poly_model.intercept_)

# 5. Smooth curve for graph
X_curve = np.linspace(1, 10, 100).reshape(-1, 1)
X_curve_poly = poly.transform(X_curve)

# 6. Plot graph
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X_curve, linear_model.predict(X_curve),
         color="red", label="Linear Regression")
plt.plot(X_curve, poly_model.predict(X_curve_poly),
         color="green", label="Polynomial Regression")

plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.grid(True)
plt.show()
