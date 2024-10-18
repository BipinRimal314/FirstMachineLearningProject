import pandas as pd

# Load the CSV file
input_file = "csv/tags.csv"
output_file = "csv/tags_cleaned.csv"

# Read the CSV into a pandas DataFrame
df = pd.read_csv(input_file)

# Ensure the 'tag' column is treated as strings, then filter out rows where 'tag' is numeric
df['tag'] = df['tag'].astype(str)  # Convert the 'tag' column to strings
df_cleaned = df[~df['tag'].str.isnumeric()]  # Filter out rows where the 'tag' is numeric

# Write the cleaned DataFrame to a new CSV file
df_cleaned.to_csv(output_file, index=False)

print(f"Cleaned data written to {output_file}")
