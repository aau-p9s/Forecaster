from darts.timeseries import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.timeseries_generation import holidays_timeseries

def generate_covariate_holidays(series: TimeSeries):
    holidays = holidays_timeseries(time_index=series.time_index, country_code="DK")
    return holidays

def generate_past_covariates(series: TimeSeries): # These are known only in the past
    weekday_series = datetime_attribute_timeseries(series, attribute="weekday", one_hot=True)
    month_series = datetime_attribute_timeseries(series, attribute="month", one_hot=True)
    holidays = generate_covariate_holidays(series)
    past_covariates = weekday_series.stack(month_series).stack(holidays)
    return past_covariates

def generate_future_covariates(series: TimeSeries): # These are known both in the future and in the past
    weekday_series = datetime_attribute_timeseries(series, attribute="weekday", one_hot=True)
    holidays = generate_covariate_holidays(series)
    future_covariates = weekday_series.stack(holidays)
    return future_covariates

# If the user knows that something unexpected happened in the past this should generate a timeseries with the incident present
def add_past_covariates(series: TimeSeries):
    pass

# If the user knows that something unexpected will happen in the future it can be added here
def add_future_covariates(series: TimeSeries):
    pass

