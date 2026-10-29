
import pandas as pd

# Read dataset
df = pd.read_excel("ai_job.xlsx")

# 1. Check dataset
print("Original Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())

# 2. Remove duplicate rows
df = df.drop_duplicates()

# 3. Fill missing categorical values with mode
for col in ["employment_type", "company_location", "company_size"]:
    df[col] = df[col].fillna(df[col].mode()[0])

# 4. Fill missing numerical values with median
df["job_description_length"] = df["job_description_length"].fillna(
    df["job_description_length"].median()
)

# 5. Check cleaned dataset
print("\nCleaned Shape:", df.shape)
print("\nRemaining Missing Values:\n", df.isnull().sum())

# 6. Save cleaned dataset
df.to_excel("ai_job_cleaned.xlsx", index=False)

print("\nCleaned dataset saved successfully!")
