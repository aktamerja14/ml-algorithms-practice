
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Read cleaned dataset
df = pd.read_excel("ai_job_cleaned.xlsx")

# Select categorical columns
data = df[["experience_level", "education_required"]]

# Create encoder
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

# Apply one-hot encoding
encoded_data = encoder.fit_transform(data)

# Convert result into DataFrame
encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoder.get_feature_names_out(data.columns)
)

# Display result
print(encoded_df.head())

print("\nOriginal shape:", data.shape)
print("Encoded shape:", encoded_df.shape)
