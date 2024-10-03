import os
import csv

from data import df
import numpy as np
from matplotlib import pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, NBEATS, LSTM
from neuralforecast.models import DilatedRNN, DLinear, NLinear, TFT 
from neuralforecast.models import HINT, TSMixer, TiDE 
from neuralforecast.models import SOFTS, DeepNPTS, TimeMixer, KAN

#OOM: All Transformer models and
#from neuralforecast.models import Informer, TFT, Autoformer, DeepNPTS

from neuralforecast.utils import AirPassengersDF

from matplotlib.patches import Rectangle

from metric import mse, mae

# normalize
MEAN, STD = df["y"].mean(), df["y"].std()
df["y"] = (df["y"] - MEAN) / STD

def forecast(input_size, h):
    
    model_names = []
    models = [
            NHITS(input_size=input_size, h=h, max_steps=1000, start_padding_enabled=True),
            #DilatedRNN(input_size=input_size, h=h, max_steps=2000),
            #NBEATS(input_size=input_size, h=h, max_steps=2000, start_padding_enabled=True),
            #DLinear(input_size=input_size, h=h, max_steps=2000, start_padding_enabled=True),
            #TiDE(input_size=input_size, h=h, max_steps=2000, start_padding_enabled=True),
            #LSTM(input_size=input_size, h=h, max_steps=2000),
            ]
    model_names.extend(["NHITS"]) #"NHITS", "DilatedRNN", "NBEATS", "DLinear", "TiDE", "LSTM"])  

    #if h < 1024:
    #    models.extend([
    #                TSMixer(input_size=input_size, h=h, n_series=1, max_steps=2000),
    #                SOFTS(input_size=input_size, h=h, n_series=1, max_steps=2000),
    #                TimeMixer(input_size=input_size, h=h, n_series=1, max_steps=2000),
    #                KAN(input_size=input_size, h=h, start_padding_enabled=True, max_steps=2000)
    #            ])
    #    model_names.extend(["TSMixer", "SOFTS", "TimeMixer", "KAN"])


    nf = NeuralForecast(models = models, freq = "M")
    
    if h > df["y"].shape[0]:
        new_df = np.zeros((h,))
        pad = h - df["y"].shape[0]
        new_df[:df["y"].shape[0]] = df["y"]
        new_df[df["y"].shape[0]:] = df["y"][:pad]
   
    print ("Training %d horizon for " % h, model_names)
    
    if all([os.path.exists("results_lookahead_%s/%d_%d.png" % (m, input_size, h)) for m in model_names]):
        return

    nf.fit(df=df)
    y_hat_forecast = nf.predict()

    for model_name in model_names:
        
        if os.path.exists("results_lookahead_%s/%d_%s_%s_%s.png" % (model_name, h, os.environ["START"], os.environ["STEP"], os.environ["LAMBDA"])):
            continue

        print ("Training %d horizon for %s" % (h, model_name))

        y_pred = y_hat_forecast[model_name]
        y_gt = df["y"][:y_pred.shape[0]]
        
        if y_pred.shape[0] > y_gt.shape[0]:
            y_gt = np.pad(y_gt, (0, y_pred.shape[0] - y_gt.shape[0]))

        l1, l2 = mse(y_pred, y_gt), mae(y_pred, y_gt)
        
        plt.plot(df["ds"], df["y"], label="training set")
        plt.plot(y_hat_forecast["ds"], y_hat_forecast[model_name], label="predicted")
        plt.legend()

        # empirically determined number of years and used unitary method
        plt.title("%s; Window Size = %d (~%.2f years)" % (model_name, input_size, (13/160)*input_size))
        if not os.path.isdir("results_lookahead_%s" % model_name):
            os.mkdir("results_lookahead_%s" % model_name)
        plt.show()
        
        print ("MSE: %.5f ; MAE: %.5f" % (l1, l2))

        plt.cla()
        plt.clf()

if __name__ == "__main__":
    
    sz = 1024
    while sz <= 1024: #1536:
        forecast(sz, sz)
        sz += 128
