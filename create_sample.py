
import pandas as pd
import os

input_path = 'data/movies.csv'
output_path = 'data/movies_sample.csv'

# Read only necessary columns to save memory if needed, or all if feasible.
# We'll stick to the ones used in training.
cols = ['id', 'title', 'genres', 'overview', 'keywords', 'credits', 'vote_count', 'vote_average', 'poster_path']

print(f"Reading {input_path}...")
df = pd.read_csv(input_path, usecols=cols)
print(f"Original shape: {df.shape}")

# Filter for relevant movies to keep quality high but size low
# Increase min_vote_count until we get a reasonable size (e.g. ~20k-30k movies)
# The training script used min_vote=50. Let's see how many rows that is.
min_votes = 100
df_filtered = df[df['vote_count'] >= min_votes].copy()

print(f"Filtered (votes >= {min_votes}) shape: {df_filtered.shape}")

# Save
print(f"Saving to {output_path}...")
df_filtered.to_csv(output_path, index=False)

file_size = os.path.getsize(output_path) / (1024 * 1024)
print(f"Sample file size: {file_size:.2f} MB")
