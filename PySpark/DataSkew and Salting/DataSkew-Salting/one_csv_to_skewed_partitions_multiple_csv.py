import pandas as pd
import numpy as np

def create_skewed_partitions(input_csv, num_partitions):
    # Read the dataset from CSV
    df = pd.read_csv(input_csv)
    
    # Determine the size of each partition
    total_rows = len(df)
    partition_sizes = [int(total_rows * (0.5 / (2 ** i))) for i in range(num_partitions)]
    
    # Adjust the last partition size to ensure all rows are included
    partition_sizes[-1] += total_rows - sum(partition_sizes)
    
    # Shuffle the dataset to ensure randomness
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Create partitions and write each to a separate CSV file
    start_idx = 0
    for i, size in enumerate(partition_sizes):
        partition = df_shuffled[start_idx:start_idx + size]
        partition.to_csv(f'partition_{i+1:02}.csv', index=False)
        start_idx += size

# Example usage
create_skewed_partitions('220k_awards_by_directors.csv', 10)