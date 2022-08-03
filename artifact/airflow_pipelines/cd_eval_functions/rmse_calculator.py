# Calculates Root Mean Squared Error for a DF and returns Series
import pandas as pd
import numpy as np


# Define RMSE function (Metric for CD evaluation)
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


# Calculates RMSE for entire DF grouped by date and returns Series (freq: "Y" -> Year, "M" -> Month, "1W" -> Week)
def rmse_calc(df, freq):
    return df.groupby(pd.Grouper(key="date", freq=freq)).apply(lambda x: rmse(x.iloc[:, 1], x.iloc[:, 2]))
