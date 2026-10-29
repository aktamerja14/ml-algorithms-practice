
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

# 1. Training data: [Study Hours, Attendance]
X = [
    [1, 30],
    [2, 35],
    [2, 45],
    [3, 40],
    [4, 50],
    [5, 60],
    [6, 65],
    [7, 75],
    [8, 85],
    [9, 90]
]

# 0 = Fail, 1 = Pass
y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

# 2. Create KNN model
model = KNeighborsClassifier(n_neighbors=3)

# 3. Train model
model.fit(X, y)

# 4. Predict for a new student
new_student = [[5, 58]]

prediction = model.predict(new_student)

print("Predicted Class:", prediction[0])

if prediction[0] == 1:
    print("Result: Pass")
else:
    print("Result: Fail")

# 5. Plot training data
plt.scatter(
    [row[0] for row in X],
    [row[1] for row in X],
    c=y,
    s=100
)

plt.scatter(
    new_student[0][0],
    new_student[0][1],
    marker="X",
    s=250,
    label="New Student"
)

plt.xlabel("Study Hours")
plt.ylabel("Attendance (%)")
plt.title("KNN Classification (K=3)")
plt.legend()
plt.grid(True)
plt.show()
