import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error


df = pd.read_csv("data/taxi_pickups_area.csv")

df.drop(columns='Trip Start Timestamp', inplace=True)

areas = df.columns.to_list()

train_size = int(len(df) * 0.8)

train = df[:train_size]
test  = df[train_size:]




from statsmodels.tsa.holtwinters import SimpleExpSmoothing

for area in areas:
    ses = SimpleExpSmoothing(train[area])

    fit = ses.fit(optimized=True)

    pred = fit.forecast(len(test[area]))


    mae = mean_absolute_error(test[area], pred)

    print(f"{area} mean absolute error using simple exp smoothing:", mae)




from statsmodels.tsa.holtwinters import ExponentialSmoothing


for area in areas:
    ses = ExponentialSmoothing(train[area], trend='add')

    fit = ses.fit(optimized=True)

    pred = fit.forecast(len(test[area]))


    mae = mean_absolute_error(test[area], pred)

    print(f"{area} mean absolute error using Holt's Exponential Smoothing trend:", mae)



for area in areas:
    ses = ExponentialSmoothing(
        train[area],
        trend='add',
        seasonal='add'
    )

    fit = ses.fit(optimized=True)

    pred = fit.forecast(len(test[area]))

    mae = mean_absolute_error(test[area], pred)

    print(f"{area} mean absolute error using Holt's Exponential Smoothing seasonality:", mae)



from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


def adf_report(x, label):
    result = adfuller(x.dropna())
    print(f"ADF test ({label}): stat={result[0]:.3f}, p-value={result[1]:.4f}")
    print("  -> Stationary" if result[1] < 0.05 else "  -> Non-stationary (differencing needed)")
 
 
print("=" * 60)
adf_report(train, "original series")
d = 0
diffed = train.copy()
while adfuller(diffed.dropna())[1] >= 0.05 and d < 3:
    diffed = diffed.diff()
    d += 1
    adf_report(diffed, f"after {d} order diff")
 
print(f"\nChosen differencing order d = {d}")


best_aic = np.inf
best_order = None
for p in range(0, 4):
    for q in range(0, 4):
        try:
            model = ARIMA(train, order=(p, d, q)).fit()
            if model.aic < best_aic:
                best_aic = model.aic
                best_order = (p, d, q)
        except Exception:
            continue
 
print(f"\nBest order by AIC search: {best_order} (AIC={best_aic:.2f})")
 
