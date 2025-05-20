import pandas as pd
import os
import glob

if __name__ == '__main__':
    directory = 'ML/Darts/Assets'
    csv_files = glob.glob(os.path.join(directory, '*.csv')) 

    rows = 0
    dataframes = []

    for file in csv_files:
        df = pd.read_csv(file)
        dataframes.append(df)
        print(f"{len(df.index)} rows in {file}")
        rows += len(df.index)

    combined_df = pd.concat(dataframes, ignore_index=True)

    combined_df.to_csv('combined_output.csv', index=False)  # Change the filename as needed
    print(f"Combined {len(csv_files)} CSV files into 'combined_output.csv' with {rows} rows.")