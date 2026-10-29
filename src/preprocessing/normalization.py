
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Read cleaned dataset
df = pd.read_excel("ai_job_cleaned.xlsx")

# Select salary column
data = df[["salary_usd"]].copy()

# Apply normalization
scaler = MinMaxScaler()
data["salary_normalized"] = scaler.fit_transform(
    data[["salary_usd"]]
)

# Display sample data
print(data.head())

# Before and after graphs
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(data["salary_usd"], kde=True, ax=axes[0])
axes[0].set_title("Before Normalization")

sns.histplot(data["salary_normalized"], kde=True, ax=axes[1])
axes[1].set_title("After Normalization")

plt.tight_layout()
plt.show()

# Check normalized range
print("\nMinimum:", data["salary_normalized"].min())
print("Maximum:", data["salary_normalized"].max())
