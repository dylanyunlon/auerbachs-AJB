# read db/R1.tbl
import os
import sys
import pandas as pd
def read_table(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return None
    try:
        df = pd.read_csv(file_path, sep='|', header=None)
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
filename = 'db/R1.tbl'
df = read_table(filename)
if df is not None:
    print(df.head())
# R1.tbl is a table with 2 columns, the edges of a graph.
# add the reverse edges to the table and delete the duplicates
if df is not None:
    df.columns = ['source', 'target']
    reverse_edges = df[['target', 'source']].rename(columns={'target': 'source', 'source': 'target'})
    combined_df = pd.concat([df, reverse_edges]).drop_duplicates().reset_index(drop=True)
    print(combined_df.head())
    combined_df.to_csv('db/edges_with_reverse.tbl', sep='|', index=False, header=False)