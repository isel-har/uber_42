# from sklearn.metrics import mean_absolute_error
# from statsmodels.tsa.seasonal import STL
# from statsmodels.tsa.arima.model import ARIMA
# from matplotlib import pyplot as plt# import plot as plt
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from utils import detect_anomalies
import pandas as pd
import numpy as np
import threading
import joblib


def transform_data(df, areas, results, idx):
    out = {}
    for area in areas:
        series = df[area].copy()
        anomalies = detect_anomalies(series, period=672)
        series.loc[series.index.isin(anomalies.index)] = 0.0

        min_val = series.min()
        shift = (abs(min_val) + 1.0) if min_val <= 0 else 0.0
        shifted_series = series + shift

        transformed_series, opt_lambda = boxcox(shifted_series)
        out[area] = transformed_series
    results[idx] = out  # each thread writes to its own slot, no lock needed




df = pd.read_csv("data/taxi_pickups_area.csv")
df['Trip Start Timestamp'] = pd.to_datetime(df['Trip Start Timestamp'])
areas = [c for c in df.columns if c != 'Trip Start Timestamp']

step = 13
areas_chunks = [areas[i:i+step] for i in range(0, len(areas), step)]

results = [None] * len(areas_chunks)
threads = [
    threading.Thread(target=transform_data, args=(df, chunk, results, i))
    for i, chunk in enumerate(areas_chunks)
]

for t in threads:
    t.start()
for t in threads:
    t.join()

# merge results back into df in the main thread only
for chunk_result in results:
    for area, values in chunk_result.items():
        df[area] = values



joblib.dump(df, "transformed_data.pkl")
print("transformed data saved")

#_________________________________________________________________________________

from sklearn.metrics import mean_absolute_error
# from statsmodels.tsa.seasonal import STL
# from statsmodels.tsa.arima.model import ARIMA
# from matplotlib import pyplot as plt# import plot as plt
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from utils import detect_anomalies
import pandas as pd
import numpy as np
import threading
import joblib


# def transform_data(df, areas, results, idx):
#     # print(f"thread of {areas} transforming...")
#     out = {}
#     results = []
#     for area in areas:
#         series = df[area].copy()
#         anomalies = detect_anomalies(series, period=672)
#         series.loc[series.index.isin(anomalies.index)] = 0.0

#         min_val = series.min()
#         shift = (abs(min_val) + 1.0) if min_val <= 0 else 0.0
#         shifted_series = series + shift

#         transformed_series, opt_lambda = boxcox(shifted_series)
#         out[area] = (transformed_series, opt_lambda)

#     results[idx] = out  # each thread writes to its own slot, no lock needed

#     return results

def transform_data(df, areas):

    out = {}
    opt_lambdas = {}

    for area in areas:
        serie = df[area].copy()
        anomalies = detect_anomalies(serie, period=672)
        serie.loc[serie.index.isin(anomalies.index)] = 0.0

        min_val = serie.min()
        shift = (abs(min_val) + 1.0) if min_val <= 0 else 0.0
        shifted_serie = serie + shift

        transformed_serie, opt_lambda = boxcox(shifted_serie)
        out[area] = transformed_serie
        opt_lambdas[area] = opt_lambda


    return out, opt_lambdas



df = pd.read_csv("data/taxi_pickups_area.csv")
df = df[df.columns.to_list()[:4]]

# areas = [c for c in df.columns if c != 'Trip Start Timestamp']

train_size = int(len(df) * 0.8)

test = df[train_size:]


transform_meta = joblib.load("transformed_meta.pkl")


transformed_df = transform_meta['transformed']
opt_lambdas = transform_meta['opt_lambda']


areas = transformed_df.columns.to_list()


averages = transformed_df.mean()

meas = pd.Series(index=areas)

forecasts = []
for area in areas:
    transformed_forecast = np.ones(len(test)) * averages[area]
    original_forecast = inv_boxcox(transformed_forecast, opt_lambdas[area])

    meas[area] = mean_absolute_error(test[area], original_forecast)

average_meas = meas.mean()

print(average_meas)
