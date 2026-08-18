# from sklearn.metrics import mean_absolute_error
# from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
# from matplotlib import pyplot as plt# import plot as plt
from scipy.stats import boxcox
from scipy.special import inv_boxcox

from utils import detect_anomalies
import pandas as pd
import numpy as np


df = pd.read_csv("data/taxi_pickups_area.csv")

areas = [c for c in df.columns.to_list() if c != 'Trip Start Timestamp']

df['Trip Start Timestamp'] = pd.to_datetime(df['Trip Start Timestamp'])


transformation_metadata = {}

for area in areas:
    series = df[area].copy()
    

    anomalies = detect_anomalies(series, period=672)

    series.loc[series.index.isin(anomalies.index)] = float(0.0)### Not sure!
    
    # series = series.interpolate(method='time').bfill().ffill()
    


    min_val = series.min()
    shift = (abs(min_val) + 1.0) if min_val <= 0 else 0.0
    shifted_series = series + shift


    transformed_series, opt_lambda = boxcox(shifted_series)
    
    df[area] = transformed_series
    

    transformation_metadata[area] = {
        'lambda': opt_lambda,
        'shift': shift
    }

print(df)


