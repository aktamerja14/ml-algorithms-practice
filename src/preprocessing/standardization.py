
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Read dataset
df = pd.read_excel("ai_job_cleaned.xlsx")

# Select numerical column
data = df[["salary_usd"]].copy()

# Apply standardization
scaler = StandardScaler()
data["salary_standardized"] = scaler.fit_transform(data[["salary_usd"]])

# Display sample data
print(data.head())

# Plot before and after standardization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(data["salary_usd"], kde=True, ax=axes[0])
axes[0].set_title("Before Standardization")

sns.histplot(data["salary_standardized"], kde=True, ax=axes[1])
axes[1].set_title("After Standardization")

plt.tight_layout()
plt.show()
