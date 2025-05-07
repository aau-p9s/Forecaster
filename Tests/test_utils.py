import pytest
import pandas as pd
from darts import TimeSeries
from ML.Darts.Utils.preprocessing import load_data, run_transformer_pipeline  # Make sure to import the function correctly from your module
import numpy as np
import re

# Create a sample CSV data for testing
@pytest.fixture
def sample_csv():
    data = {
        'timestamp': ['2025-03-10 00:01', '2025-03-10 00:02', '2025-03-10 00:03'],
        'value': [10, 20, 30]
    }
    df = pd.DataFrame(data)
    # Save to a temporary CSV file for testing
    file_path = 'test_data.csv'
    df.to_csv(file_path, index=False)
    return file_path

@pytest.fixture
def sample_data():
    data = [
        ['2025-03-10 00:01', 1.3609763251569478],
        ['2025-03-10 00:02', 1.375],
        ['2025-03-10 00:03', 1.390],
        ['2025-03-10 00:04', 1.405],
        ['2025-03-10 00:05', 1.420],
        ['2025-03-10 00:06', 1.435],
        ['2025-03-10 00:07', 1.450],
        ['2025-03-10 00:08', 1.465],
        ['2025-03-10 00:09', 1.480],
        ['2025-03-10 00:10', 1.495],
        ['2025-03-10 00:11', 1.510]
    ]
    return data
@pytest.fixture
def sample_data_epoch():
    data = [
        [1741105777.945, 1.3609763251569478],
        [1741105837.945, 1.375],
        [1741105897.945, 1.390],
        [1741105957.945, 1.405],
        [1741106017.945, 1.420],
        [1741106077.945, 1.435],
        [1741106137.945, 1.450],
        [1741106197.945, 1.465],
        [1741106257.945, 1.480],
        [1741106317.945, 1.495],
        [1741106377.945, 1.510],
    ]
    return data

@pytest.fixture
def sample_timeseries_missing_values():
    data = {
        'timestamp': ['2025-03-10 00:01', '2025-03-10 00:02', '2025-03-10 00:03'],
        'value': [10, np.nan, 30]
    }
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    return TimeSeries.from_dataframe(df, time_col='timestamp', value_cols=['value'])


def test_load_data_with_infered_granularity_from_csv(sample_csv):
    granularity = 'min'
    
    ts = load_data(sample_csv)
    
    assert isinstance(ts, TimeSeries)
    
    # This checks if freq is inferred correctly
    assert ts.freq == granularity, f"Expected granularity {granularity}, but got {ts.freq}"

def test_load_data_with_infered_granularity_from_json(sample_data):
    granularity = 'min'
    
    ts = load_data(sample_data)
    
    assert isinstance(ts, TimeSeries)
    
    # This checks if freq is inferred correctly
    assert ts.freq == granularity, f"Expected granularity {granularity}, but got {ts.freq}"

def test_load_data_with_infered_granularity_from_epoch(sample_data_epoch):
    granularity = 'min'
    
    ts = load_data(sample_data_epoch)
    
    assert isinstance(ts, TimeSeries)
    
    # This checks if freq is inferred correctly
    assert ts.freq == granularity, f"Expected granularity {granularity}, but got {ts.freq}"

def test_transformer_pipeline_with_missing_values(sample_timeseries_missing_values : TimeSeries):

    assert sample_timeseries_missing_values.to_dataframe().isna().any().any()

    ts, ratio = run_transformer_pipeline(sample_timeseries_missing_values)

    assert not ts.to_dataframe().isna().any().any()
    assert isinstance(ratio, float)

    df_processed = ts.pd_dataframe()

    assert df_processed.values.min() >= 0, "Found values below 0, scaling failed"
    assert df_processed.values.max() <= 1, "Found values above 1, scaling failed"
