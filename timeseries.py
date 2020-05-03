## libraries
import numpy as np
import pandas as pd

## models
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from fbprophet import Prophet


def test_stationary(timeseries):
    """
    Perform Dickey-Fuller Test: determine if timeseries is stationary.
    """

    ## compute rolling statistics
    rolling_mean = timeseries.rolling(30).mean()
    rollwing_std = timeseries.rolling(30).std()

    # plot rolling statistics
    fig, ax = plt.subplots(figsize=(16,6))
    original = ax.plot(timeseries, color="dodgerblue", label="Original")
    mean = ax.plot(rolling_mean, color="crimson", label="Rolling Mean (30)")
    std = ax.plot(rollwing_std, color="black", label="Rolling Std (30)")
    ax.legend(loc="best")
    ax.set_title("Rolling Statistics")
    plt.show(block=False)

    ## perform test
    print('***** Dickey-Fuller Test *****')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

def log_shift(data, shift):
    """
    Perform log data shift.
    Input array (data) and return the shifted log of data.
    Return log of data and shift of log data.
    """

    ## compute log data
    ts_log = np.log(data)

    ## compute difference between data and shift
    ts_log_diff = ts_log - ts_log.shift(shift)

    ## eliminate null
    ts_log_diff = ts_log_diff.dropna()

    ## create Series to be returned
    ts_log_diff = pd.Series(ts_log_diff.values ,index=ts_log_diff.index)

    ## return log of data and shift of log data
    return ts_log, ts_log_diff

def decompose(data, col):

    """
    Decompose data into a trend, a seasonality, and a residual.
    Plot the decomposition.
    """

    ## compute log of data and shift of log data
    ts_log, ts_log_diff = log_shift(data[col], 1)

    ## decompose data based on seasonality
    ## decomposition contains trend, seasonality, and residual
    decomposition = seasonal_decompose(ts_log)

    ## unpack decomposition
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    ## plot results
    fig, axes = plt.subplots(4,1,figsize=(16,18))

    axes[0].plot(ts_log, label="Log Original")
    axes[0].legend(loc="best")
    axes[1].plot(trend, label="Trend")
    axes[1].legend(loc="best")
    axes[2].plot(seasonal, label="Seasonality")
    axes[2].legend(loc="best")
    axes[3].plot(residual, label="Residuals")
    axes[3].legend(loc="best")
    plt.tight_layout()

    ## return decomposition
    return trend, seasonal, residual

def transform(data, window):
    """
    Remove trend from log data.
    The trend is defined as the rolling mean.
    Input: data as array, window as integer
    Output: transformed data as array
    """
    ## log of data
    ts_log = np.log(data)

    ## compute trend = rolling mean
    avg_log = ts_log.rolling(window= window).mean()

    ## remove trend from data
    diff_ts_avg = (ts_log - avg_log).dropna()
    
    ## return transformed data
    return diff_ts_avg

def acf_pacf(timeseries):
    """
    Plot ACF (Auto-Correlation Function) and PACF (Partial Auto-Correlation Function).

    A time series can have components like trend, seasonality, cyclic and residual.
    ACF considers all these components while finding correlations hence it’s a ‘complete auto-correlation plot’.

    PACF is a partial auto-correlation function. Basically instead of finding correlations of present with lags
    like ACF, it finds correlation of the residuals (which remains after removing the effects which are already
    explained by the earlier lag(s)) with the next lag value hence ‘partial’ and not ‘complete’ as we remove
    already found variations before we find the next correlation.
    """

    ## create plot components
    fig, axes = plt.subplots(1,2,figsize=(16,6))

    ## compute acf and pcf
    ts_log_diff = transform(timeseries, 30)
    lag_acf = acf(ts_log_diff, nlags=10)
    lag_pacf = pacf(ts_log_diff, nlags=10, method="ols")

    ## plot ACF
    axes[0].plot(lag_acf, color="dodgerblue")
    axes[0].axhline(y=0,linestyle='--',color='green')
    axes[0].axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='crimson')
    axes[0].axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='crimson')
    axes[0].set_title('Autocorrelation Function')

    ## plot PACF
    axes[1].plot(lag_acf, color="dodgerblue")
    axes[1].axhline(y=0,linestyle='--',color='green')
    axes[1].axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='crimson')
    axes[1].axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='crimson')
    axes[1].set_title('Autocorrelation Function')

    plt.tight_layout()

def arima_funtion(ts_log, ts_log_diff, order, plot=False):

    """
    Fit an ARIMA model on the ts_log.
    Return fitted values.
    Assumptions of the model:
        - Stationary for AR models
        - Invertibility for MA models
    """

    ## create model
    model = ARIMA(ts_log, order=order)

    ## fit model
    results = model.fit(disp=-1) 

    ## compute metric of interest (MSE)
    rss = np.sum((results.fittedvalues-ts_log_diff)**2)

    ## plot results
    if plot:
        fig, ax = plt.subplots(figsize=(16,6))
        ax.plot(ts_log_diff, color="dodgerblue")
        ax.plot(results.fittedvalues, color="crimson")
        ax.set_title('RSS: %.4f'% rss)

    ## store results
    fitted = results.fittedvalues()

    ## print metrics
    print("RSS of the model is = " + str(rss))
    return fitted

def fit_arima(ts, order, plot=False):
    """
    Create an ARIMA model based on log shifting the data by 1 unit.
    Returns fitted model.
    """

    ## compute log and diff log
    ts_log, ts_log_diff = log_shift(ts, 1)

    ## build model
    model = ARIMA(ts_log, order=(1,1,1))

    ## make predictions
    results = model.fit(disp=-1)

    ## plot
    if plot:
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(results.fittedvalues, color="crimson")
        ax.set_title("RSS: {:.4f}".format(sum((results.fittedvalues-ts_log_diff)**2)))
    
    ## return results
    return results

def predict_arima(ts, order, plot=False):
    """
    Returns the predicted values from ARIMA model
    """
    ## create and fit model
    results_ARIMA = fit_arima(ts, order, plot)

    ## store result in pd.Series
    predictions_ARIMA = pd.Series(results_ARIMA.fittedvalues, copy=True)

    ## return predictions
    return predictions_ARIMA

def make_continuous(df):
    """
    Transform the data as continuous to be compatible with the seasonal_decompose function
    """
    ## change index to date
    df = df[['price', 'Date']].set_index('Date')

    ## set the frequency as 1 Day
    df = df.asfreq(freq='1D')
    
    ## fill missing values by interpolating
    df['price'].interpolate(inplace = True)
    
    ## return results
    return df

def prophet_forecast(data, period, changepoint_prior_scale=0.5, plot=False):
    '''
    Uses make_continuos func to convert intermittent data into
    a continuous one and then fits fb prophet time-series model
    Args: data and period
    Return: forecasts with graph, Monthly and weekly trend
    '''

    ## transform data
    data = make_continuous(data)

    ## storage
    df = pd.DataFrame()
    df['ds'] = data.index
    df['y'] = data.price.values

    ## create model
    m = Prophet(changepoint_prior_scale=changepoint_prior_scale)

    ## fit model
    m_fit = m.fit(df)

    ## make predictions
    future = m.make_future_dataframe(periods = period)
    forecast = m.predict(future)
    forecast = forecast.round(0)

    ## plot
    if plot:
        fig1 = m.plot(forecast)
        fig2 = m.plot_components(forecast)

    return forecast