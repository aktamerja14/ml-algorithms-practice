
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read cleaned dataset
df = pd.read_excel("ai_job_cleaned.xlsx")

# Create 3 graphs in one figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Graph 1: Job count by experience level
sns.countplot(data=df, x="experience_level", ax=axes[0])
axes[0].set_title("Job Count by Experience Level")
axes[0].set_xlabel("Experience Level")
axes[0].set_ylabel("Job Count")

# Graph 2: Average salary by experience level
sns.barplot(
    data=df, x="experience_level",
    y="salary_usd", errorbar=None, ax=axes[1]
)
axes[1].set_title("Average Salary by Experience Level")
axes[1].set_xlabel("Experience Level")
axes[1].set_ylabel("Average Salary (USD)")

# Graph 3: Salary distribution
sns.histplot(data=df, x="salary_usd", kde=True, ax=axes[2])
axes[2].set_title("Salary Distribution")
axes[2].set_xlabel("Salary (USD)")
axes[2].set_ylabel("Frequency")

# Adjust layout and show all graphs
plt.tight_layout()
plt.show()
