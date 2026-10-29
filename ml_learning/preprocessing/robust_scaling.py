
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler

# Read cleaned dataset
df = pd.read_excel("ai_job_cleaned.xlsx")

# Select salary column
data = df[["salary_usd"]].copy()

# Apply Robust Scaling
scaler = RobustScaler()
data["salary_robust"] = scaler.fit_transform(
    data[["salary_usd"]]
)

# Display sample data
print(data.head())

# Before and after graphs
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.boxplot(x=data["salary_usd"], ax=axes[0])
axes[0].set_title("Before Robust Scaling")

sns.boxplot(x=data["salary_robust"], ax=axes[1])
axes[1].set_title("After Robust Scaling")

plt.tight_layout()
plt.show()
