
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Read cleaned dataset
df = pd.read_excel("ai_job_cleaned.xlsx")

# Select categorical columns
columns = ["experience_level", "education_required"]

# Create encoder
encoder = OrdinalEncoder(categories=[
    ["EN", "MI", "SE", "EX"],
    ["Associate", "Bachelor", "Master", "PhD"]
])

# Apply ordinal encoding
df[columns] = encoder.fit_transform(df[columns])

# Display result
print(df[columns].head())
