
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

# 1. Dataset: Study hours and attendance
X = np.array([
    [1, 30], [2, 35], [2, 45], [3, 40],
    [3, 50], [4, 45], [4, 55], [5, 50],
    [5, 60], [6, 65], [6, 75], [7, 70],
    [7, 80], [8, 75], [8, 90], [9, 85]
])

# 0 = Fail, 1 = Pass
y = np.array([
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1
])

# 2. Train model
model = LogisticRegression()
model.fit(X, y)

# 3. Create grid for decision regions
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 5, X[:, 1].max() + 5

xx, yy = np.meshgrid(
    np.linspace(x1_min, x1_max, 200),
    np.linspace(x2_min, x2_max, 200)
)

grid = np.c_[xx.ravel(), yy.ravel()]

# Probability of class 1
probability = model.predict_proba(grid)[:, 1]
probability = probability.reshape(xx.shape)

# 4. Plot positive and negative regions
plt.contourf(
    xx, yy, probability,
    levels=[0, 0.5, 1],
    alpha=0.3
)

# Decision boundary at probability 0.5
plt.contour(
    xx, yy, probability,
    levels=[0.5],
    colors="black"
)

plt.scatter(
    X[:, 0], X[:, 1],
    c=y, edgecolors="black"
)

plt.xlabel("Study Hours")
plt.ylabel("Attendance (%)")
plt.title("Logistic Regression Decision Boundary")
plt.show()

# 5. Predict for a new student
student = np.array([[5, 65]])

predicted_class = model.predict(student)[0]
probability_pass = model.predict_proba(student)[0, 1]

print("Predicted Class:", predicted_class)
print("Probability of Pass:", round(probability_pass, 3))

if predicted_class == 1:
    print("Result: Pass")
else:
    print("Result: Fail")
