import pandas as pd
import numpy as np

# Load your existing dataset
df_existing = pd.read_csv()

# Function to generate synthetic data based on the existing dataset
def generate_synthetic_data(existing_data, num_samples):
    synthetic_data = pd.DataFrame()
    
    # Repeat the generation process until reaching the desired number of samples
    while len(synthetic_data) < num_samples:
        # Example: Randomly shuffle the rows
        new_synthetic_data = existing_data.sample(frac=1).reset_index(drop=True)
        
        # Append the new synthetic data to the existing synthetic data
        synthetic_data = pd.concat([synthetic_data, new_synthetic_data], ignore_index=True)
    
    # Trim the excess rows to exactly match the desired number of samples
    synthetic_data = synthetic_data.head(num_samples)
    
    return synthetic_data

# Generate 2000 synthetic data
num_samples = 2000
synthetic_data = generate_synthetic_data(df_existing, num_samples)

# Save synthetic data to a new CSV file
synthetic_data.to_csv('Synthetic_Dataset_3000_rows_with_reference.csv', index=False)