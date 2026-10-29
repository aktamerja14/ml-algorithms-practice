
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# Sample dataset
X = pd.DataFrame({
    "hours_studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "attendance": [40, 45, 50, 55, 65, 70, 75, 85, 90, 95]
})

y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Train Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predict
y_pred2 = dt_model.predict(X_test)

# Metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred2, labels=[0, 1]))

print("Precision:", precision_score(y_test, y_pred2, zero_division=0))
print("Recall:", recall_score(y_test, y_pred2, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred2, zero_division=0))

print("Precision for each class:",
      precision_score(y_test, y_pred2, average=None, zero_division=0))

print("Recall for each class:",
      recall_score(y_test, y_pred2, average=None, zero_division=0))
