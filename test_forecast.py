from darts.models import StatsForecastAutoTheta
import matplotlib.pyplot as plt
from darts.metrics import rmse
from argparse import ArgumentParser
import ML.Darts.Utils.preprocessing

if __name__ == "__main__":
    parser = ArgumentParser(description="Test model")
    parser.add_argument("-o", "--output", action="store_true", help="Output folder")
    parser.add_argument("-i", "--input", help="Input data")

    parsed_args = parser.parse_args()
    input = parsed_args.input

    series = ML.Darts.Utils.preprocessing.load_data(input, "H")
    series, missing_values_ratio = (
        # ML.Darts.Utils.preprocessing.run_transformer_pipeline(series)
        ML.Darts.Utils.preprocessing.handle_missing_values(series)
    )
    print(f"Ratio of missing values: {missing_values_ratio}\n")
    print(len(series))
    train_series, val_series = series.split_after(0.75)
    forecast_period = 48

    model = StatsForecastAutoTheta(season_length=48)
    print("Training...")
    model.fit(train_series)
    print("Predicting...")
    forecast = model.predict(forecast_period)
    val_target = val_series[
        :forecast_period
    ]  # This is used to ensure that the forecast is validated against the ground truth for the same timeframe
    rmse_value = rmse(val_target, forecast)
    print(rmse_value)
    plt.figure()
    train_series[-len(forecast) :].plot(label="Train")
    val_series[: len(forecast)].plot(label="Actual")
    forecast.plot(label="Forecast")
    plt.legend()
    plt.grid(True)
    plt.savefig("TestTheta.png")
