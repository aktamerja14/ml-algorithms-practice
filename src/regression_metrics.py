
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Actual and predicted values
y_actual = [50000, 60000, 70000, 80000, 90000]
y_pred = [52000, 58000, 72000, 75000, 95000]

# Calculate metrics
mae = mean_absolute_error(y_actual, y_pred)
mse = mean_squared_error(y_actual, y_pred)
r2 = r2_score(y_actual, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", mse ** 0.5)
print("R2 Score:", r2)
