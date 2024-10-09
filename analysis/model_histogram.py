import os

#from trend import df_increasing as df

import numpy as np
from matplotlib import pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, NBEATS, LSTM
from neuralforecast.models import DilatedRNN, DLinear, NLinear, TFT 
from neuralforecast.models import HINT, TSMixer, TiDE 
from neuralforecast.models import SOFTS, DeepNPTS, TimeMixer, KAN

#OOM: All Transformer models and
#from neuralforecast.models import Informer, TFT, Autoformer, DeepNPTS

from data import create_uniform_data
df = create_uniform_data(data_size=1024, spacing=100)
# normalize
MEAN, STD = df["y"].mean(), df["y"].std()
df["y"] = (df["y"] - MEAN) / STD

h = 1024
if h > df["y"].shape[0]:
    df = create_uniform_data(data_size=h, spacing=100)
    # normalize
    MEAN, STD = df["y"].mean(), df["y"].std()
    df["y"] = (df["y"] - MEAN) / STD

START, STEP, LAMBDA = os.environ["START"], os.environ["STEP"], os.environ["LAMBDA"]
os.environ["START"] = "1"
os.environ["STEP"] = "1"
os.environ["LAMBDA"] = "0.5"

models = [
        NHITS(input_size=h, h=h, max_steps=2000, start_padding_enabled=True),
        #DilatedRNN(input_size=input_size, h=h, max_steps=2000),
        NBEATS(input_size=h, h=h, max_steps=2000, start_padding_enabled=True),
        DLinear(input_size=h, h=h, max_steps=2000, start_padding_enabled=True),
        #TiDE(input_size=input_size, h=h, max_steps=2000, start_padding_enabled=True),
        #LSTM(input_size=input_size, h=h, max_steps=2),
        ]
model_names = ["NHITS", "NBEATS", "DLinear"] #, "TiDE"]) #"DilatedRNN", "DLinear", "LSTM"])  

nf = NeuralForecast(models = models, freq = "M")

nf.fit(df=df)
y_hat_forecast_sota = nf.predict()

os.environ["START"] = START
os.environ["STEP"] = STEP
os.environ["LAMBDA"] = LAMBDA

nf.fit(df=df)
y_hat_forecast_pred_self_sup = nf.predict()

for model_name in model_names:
    
    y_pred_sota = np.array(y_hat_forecast_sota[model_name])
    y_gt = np.array(df["y"][:y_pred_sota.shape[0]])
    
    err_sota = np.abs(y_pred_sota - y_gt)

    y_pred_self_sup = np.array(y_hat_forecast_pred_self_sup[model_name])
    err_pred_self_sup = np.abs(y_pred_self_sup - y_gt)
    
    x = np.linspace(0, 1, 100)
    
    fig, (ax1,ax2) = plt.subplots(nrows=2, sharex=True)

    extent = [0, h, 0, 1]
    
    mappable1 = ax1.imshow(err_sota[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)
    ax1.set_yticks([])
    ax1.set_xlim(extent[0], extent[1])
    ax1.set_xlabel("%s model error (MAE=%.5f)" % (model_name, np.mean(err_sota)))

    mappable2 = ax2.imshow(err_pred_self_sup[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)
    ax2.set_yticks([])
    ax2.set_xlim(extent[0], extent[1])
    ax2.set_xlabel("%s prediction self-supervised model error (MAE=%.5f)" % (model_name, np.mean(err_pred_self_sup)))
    
    plt.colorbar(mappable=mappable1)
    plt.colorbar(mappable=mappable2)

    #plt.legend(loc='upper right')
    #plt.show()
    plt.savefig("results_lookahead_%s/%d_histogram.png" % (model_name, h))
