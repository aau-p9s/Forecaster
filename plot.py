import pandas as pd
import matplotlib.pyplot as plt


def main():
    # Path to your CSV file
    filepath = "../Datasets/ServerRequest1.csv"

    # Load the CSV
    df = pd.read_csv(filepath)

    # Assume the first column is time, second column is values
    time_col = df.columns[0]
    value_col = df.columns[1]

    # Parse time column (if needed)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    # Plot
    plt.plot(df[time_col][:11520], df[value_col][:11520])
    plt.title(f"Plot of {value_col}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.savefig("DataB4scaling.png")


if __name__ == "__main__":
    main()
