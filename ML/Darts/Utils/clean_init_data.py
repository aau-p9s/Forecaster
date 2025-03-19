import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', "--input")
args = parser.parse_args()
file_name = str(args.input).split(".")[0]
df = pd.read_excel(args.input, skiprows=11, names=["timestamp", "requests"])

output_file = file_name + ".csv"
df.to_csv(output_file, index=False, mode="w")
print(f"Cleaned data saved to {output_file}")