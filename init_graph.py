import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file
def plot_csv(filename, y_column=None):
    df = pd.read_csv(filename, parse_dates=[0])  # Parse first column as timestamps
    df.set_index(df.columns[0], inplace=True)  # Set first column as index
    
    # If no specific y-column is provided, use the first numerical one
    if y_column is None:
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            raise ValueError("CSV must have at least one numerical column to plot.")
        y_column = numeric_cols[0]
    
    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df[y_column], marker='o', linestyle='-')
    plt.xlabel("Timestamp")
    plt.ylabel(y_column)
    plt.title(f"Plot of {y_column} over time")
    plt.grid()
    plt.savefig("ServerRequest3.png")
    plt.show()

if __name__ == "__main__":
    plot_csv("ML\Darts\Assets\ServerRequest3.csv")