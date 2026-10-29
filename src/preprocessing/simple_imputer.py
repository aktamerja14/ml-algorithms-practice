
import pandas as pd
from sklearn.impute import SimpleImputer

# Read dataset
df = pd.read_excel("ai_job.xlsx")

# Check missing values before imputation
print("Before Imputation:")
print(df.isnull().sum())

# Select numerical column
imputer = SimpleImputer(strategy="median")

df[["job_description_length"]] = imputer.fit_transform(
    df[["job_description_length"]]
)

# Check missing values after imputation
print("\nAfter Imputation:")
print(df.isnull().sum())
