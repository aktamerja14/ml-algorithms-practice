
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

# 1. Dataset
X = np.array([
    [1], [2], [3], [4], [5],
    [6], [7], [8], [9], [10]
])

# 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# 2. Train model
model = LogisticRegression()
model.fit(X, y)

# 3. Predict for a student studying 4 hours
hours = np.array([[4]])

prediction = model.predict(hours)
probability = model.predict_proba(hours)

print("Predicted Class:", prediction[0])
print("Probability of Fail:", round(probability[0][0], 3))
print("Probability of Pass:", round(probability[0][1], 3))

if prediction[0] == 1:
    print("Result: Pass")
else:
    print("Result: Fail")

# 4. Create points for the graph
X_graph = np.linspace(0, 11, 200).reshape(-1, 1)
prob_pass = model.predict_proba(X_graph)[:, 1]

# 5. Plot
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X_graph, prob_pass, color="red",
         label="Pass Probability")

plt.axhline(0.5, color="green", linestyle="--",
            label="Threshold = 0.5")

plt.xlabel("Study Hours")
plt.ylabel("Probability of Pass")
plt.title("Logistic Regression")
plt.ylim(-0.1, 1.1)
plt.legend()
plt.grid(True)
plt.show()
