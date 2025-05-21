import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from ML.Darts.Utils.preprocessing import run_transformer_pipeline


def main():
    filepath = "ServerRequest1.csv"
    timeseries = TimeSeries.from_csv(filepath, time_col='timestamp', value_cols='requests', freq="min")

    # Store original before transformation
    original = timeseries

    # Run preprocessing
    transformed, missing, _ = run_transformer_pipeline(timeseries)

    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    original.plot(ax=axes[0])
    axes[0].set_title("Original Time Series")

    transformed.plot(ax=axes[1])
    axes[1].set_title("Transformed Time Series")

    print(f"Missing value ratio: {missing}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()