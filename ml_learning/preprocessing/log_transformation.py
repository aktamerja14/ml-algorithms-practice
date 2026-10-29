
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read cleaned dataset
df = pd.read_excel("ai_job_cleaned.xlsx")

# Select salary column
data = df[["salary_usd"]].copy()

# Apply log transformation
data["salary_log"] = np.log1p(data["salary_usd"])

# Display sample data
print(data.head())

# Before and after graphs
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(data["salary_usd"], kde=True, ax=axes[0])
axes[0].set_title("Before Log Transformation")

sns.histplot(data["salary_log"], kde=True, ax=axes[1])
axes[1].set_title("After Log Transformation")

plt.tight_layout()
plt.show()
