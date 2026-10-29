
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Read dataset
df = pd.read_excel("ai_job_cleaned.xlsx")

# Select input and target
X = df[["years_experience"]]
y = df["salary_usd"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict salary
y_pred = model.predict(X_test)

# Display results
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Plot actual data and regression line
plt.scatter(X_test["years_experience"], y_test, label="Actual")
plt.scatter(X_test["years_experience"], y_pred, label="Predicted")

plt.xlabel("Years of Experience")
plt.ylabel("Salary (USD)")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()
