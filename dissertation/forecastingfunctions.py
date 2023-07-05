
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

def plot_acf_values(data, lags):
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_acf(data.dropna(), lags=lags, ax=ax,zero=False)
    plt.title('ACF')
    plt.show()

def plot_pacf_values(data, lags):
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_pacf(data.dropna(), lags=lags, ax=ax,zero=False,alpha=0.05)
    plt.title('PACF')
    plt.show()

def find_auto_arima(dataframe,column,exogcolumn=None):
  model = auto_arima(dataframe[column].dropna(),
                exog=exogcolumn,
                   start_p=1, start_q=1,
                   max_p=5, max_q=5, m=0,
                   seasonal=False,start_d=0,
                  
                   error_action='ignore',
                   suppress_warnings=True,
                   stepwise=True)
  return model
  
def arima_model_fit(dataframe,columnname,p,d,q,exog=None):
  model=ARIMA(dataframe[columnname].dropna(),order=(p,d,q))
  model_fit=model.fit()
  return model_fit

def Transform_OriginalValue(df,PredictedValue,Number_pct_change,original):
    initial_value = df[original][0]
    df['reversediff']=df[Number_pct_change].shift(1) + df[PredictedValue]
    initial_value = df[original][0]
    df.dropna(subset=[Number_pct_change], inplace=True)
    first_date = df.index[0]
    df.loc[df.index == first_date, 'reversediff'] = df.loc[df.index == first_date, Number_pct_change]
    df[original+'Predicted'] = (df['reversediff'] + 1).cumprod() * initial_value
    df = df.drop('reversediff', axis=1)

from statsmodels.tsa.stattools import adfuller

def perform_adfuller(series):
    # Perform Augmented Dickey-Fuller test
    result = adfuller(series)

    # Extract test statistics and p-value
    test_statistic = result[0]
    p_value = result[1]

    # Print the results
    print("Augmented Dickey-Fuller Test:")
    print(f"Test Statistic: {test_statistic}")
    print(f"P-value: {p_value}")

    # Check the p-value against a significance level (e.g., 0.05) to determine stationarity
    if p_value <= 0.05:
        print("The time series is stationary.")
    else:
        print("The time series is non-stationary.")


print("All Executed")