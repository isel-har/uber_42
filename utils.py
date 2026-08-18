from statsmodels.tsa.seasonal import STL

def detect_anomalies(serie, robust=True, period=672):

    stl = STL(serie, robust=robust, period=period)

    result = stl.fit()

    resid =  result.resid

    resid_m = resid.mean()
    resid_dv = resid.std()

    lower = resid_m - (3 * resid_dv) 
    upper = resid_m + (3 * resid_dv)

    return serie[(resid < lower) | (resid > upper)]



# from sklearn.base import BaseEstimator, TransformerMixin

# class AnomaliesFilter(BaseEstimator, TransformerMixin):

#     def __init__(self, robust=True, period=672):
#         self.robust = robust
#         self.period = period

#     def fit(self, df, columns=[]):

#         for col in columns:

#             serie = df[col]

#             anomalies = detect_anomalies(serie, robust=self.robust, period=self.period)

#             filtered_serie = serie[~serie.index.isin(anomalies.index)]
#             df[col] = filtered_serie

#         return df





