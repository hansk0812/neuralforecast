import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def create_uniform_data(data_size=1000, spacing=100):
    # multiples of 100 only

    np.random.seed(161294)
    x = np.arange(0, spacing, spacing / data_size)
    y = np.zeros_like(x)

    for idx in range(data_size // spacing):
        y[idx*spacing : idx*spacing + spacing] = signal.windows.gaussian(spacing, std=1) * 10 #* np.random.rand()

    df = pd.DataFrame()
    df["unique_id"] = [1. for _ in range(len(y))]
    df["y"] = y

    base = datetime.today()
    date_list = [base - relativedelta(months=idx) for idx in range(data_size)]
    df["ds"] = date_list
    
    return df

if __name__ == "__main__":
    df = create_uniform_data(1024, 100)
    
    df["y"].plot()
    plt.show()

    print (df)
    print ('min', df["y"].min(), 'max', df["y"].max())
