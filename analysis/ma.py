import numpy as np
from matplotlib import pyplot as plt

#from data import df
from trend import df_decreasing as df

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Trend
def moving_average(a, n=3):
    ret = np.cumsum(np.array(a), dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# Trend
def moving_average_conv(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

if __name__ == "__main__":
    
    N = 20

    # Multiplicative decomposition isn't appropriate for zeros and negative values
    df["y"] = df["y"].abs() +1e-5
    
    y_out = np.zeros_like(df["y"])
    y_out_conv = np.zeros_like(df["y"])
    y_out[N//2-1:-N//2] = moving_average(df["y"], n=N)
    y_out_conv[N//2-1:-N//2] = moving_average_conv(df["y"], w=N)
    plt.plot(df["y"])
    plt.plot(y_out)
    plt.plot(y_out_conv)
    plt.show()

    #...........................................................................#

    result = seasonal_decompose(df["y"], period=30, model='multiplicative')
    result.plot()
    plt.show()
    
    result = seasonal_decompose(df["y"], period=30, model='additive')
    result.plot()
    plt.show()
    #...........................................................................#

    ema_window = 30  # 30-day moving average
    ema = df["y"].ewm(span=ema_window, adjust=False).mean()
    
    result = seasonal_decompose(df["y"], period=1, model='multiplicative')
    result.plot()
    plt.show()
    
    result = seasonal_decompose(df["y"], period=1, model='additive')
    result.plot()
    plt.show()   
    #...........................................................................#

    E = ExponentialSmoothing(df["y"], damped_trend=False, seasonal="additive", seasonal_periods=100)
    res = E.fit()
    pred_A = E.predict(res.params, start=1001, end=1500)

    E = ExponentialSmoothing(df["y"], damped_trend=False, seasonal="multiplicative", seasonal_periods=100)
    res = E.fit()
    pred_M = E.predict(res.params, start=1001, end=1500)

    plt.plot(df["y"])
    plt.plot(pred_A)
    plt.plot(pred_M)
    plt.show()
