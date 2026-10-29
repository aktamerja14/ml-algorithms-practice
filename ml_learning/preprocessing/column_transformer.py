
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Read cleaned dataset
df = pd.read_excel("ai_job_cleaned.xlsx")

# Select input columns
X = df[
    ["years_experience", "salary_usd",
     "experience_level", "employment_type"]
].copy()

# Numerical and categorical columns
numerical_cols = ["years_experience", "salary_usd"]
categorical_cols = ["experience_level", "employment_type"]

# Create Column Transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# Apply preprocessing
X_transformed = preprocessor.fit_transform(X)

# Convert output into DataFrame
columns = preprocessor.get_feature_names_out()

X_transformed = pd.DataFrame(
    X_transformed,
    columns=columns,
    index=X.index
)

# Display result
print("Before preprocessing:")
print(X.head())

print("\nAfter preprocessing:")
print(X_transformed.head())

print("\nOriginal shape:", X.shape)
print("Transformed shape:", X_transformed.shape)
