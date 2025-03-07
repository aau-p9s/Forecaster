from darts import TimeSeries
import pandas as pd
def load_data(data_path, granularity):
        """
        Args:
            data_path (str): Path to csv
            granularity (str): The interval between each timestamp. Must be one of these: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        """
        df = pd.read_csv(data_path)
        ts = TimeSeries.from_dataframe(df, time_col=df.columns[0], value_cols=df.columns[1:], freq=granularity)
        return ts