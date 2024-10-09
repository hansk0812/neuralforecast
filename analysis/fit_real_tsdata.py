import os

from matplotlib import pyplot as plt

from datasetsforecast.utils import Info
from datasetsforecast.long_horizon import LongHorizon 
from datasetsforecast.long_horizon import ETTh1, ETTh2, ETTm1, ETTm2, \
                                            ECL, Exchange, TrafficL, ILI, Weather

from neuralforecast.models import NHITS
from neuralforecast import NeuralForecast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pytorch_lightning.callbacks import ModelCheckpoint

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--model_path", help="Path to model to load from", default=None)
args = ap.parse_args()

h = 32

OUTLIER_THRESHOLDS = [
                        [0,0], 
                        [0,0],
                        [0,0],
                        [-6,-4.25],
                        [20,15.3],
                        [0,0],
                        [0,0],
                        [0,0],
                        [-150,-30]
                     ]

TIME_INTERVALS = [
                    "H",
                    "H",
                    "15m",
                    "15m",
                    "1h",
                    "1d",
                    "1h",
                    "1w",
                    "10m"
                 ]

def remove_outliers(data, idx):

    if OUTLIER_THRESHOLDS[idx][0] == OUTLIER_THRESHOLDS[idx][1] and OUTLIER_THRESHOLDS[idx][0] == 0:
        return
    
    if OUTLIER_THRESHOLDS[idx][0] < 0:
        data["y"][data["y"] < OUTLIER_THRESHOLDS[idx][0]] = OUTLIER_THRESHOLDS[idx][1]
    else:
        data["y"][data["y"] > OUTLIER_THRESHOLDS[idx][0]] = OUTLIER_THRESHOLDS[idx][1]
    
LongHorizonInfo = Info((
        ETTh1, ETTh2, ETTm1, ETTm2, 
        ECL, Exchange, TrafficL, ILI, Weather
))

for idx, (group, meta) in enumerate(LongHorizonInfo):
    data, *_ = LongHorizon.load(directory='data', group=group)
    unique_elements = data.groupby(['unique_id', 'ds']).size()
    unique_ts = data.groupby('unique_id').size()

    assert (unique_elements != 1).sum() == 0, f'Duplicated records found: {group}'
    assert unique_ts.shape[0] == meta.n_ts, f'Number of time series not match: {group}'
    
    data["y"] = (data["y"] - data["y"].mean()) / data["y"].std()

    #data["y"].plot()
    #plt.show()
    remove_outliers(data, idx)
    
    #data["y"].plot()
    #plt.show()
    #print (data["ds"])
    
    data["ds"] = pd.to_datetime(data["ds"])
    
    L = len(data)
#    train_split = int(L * 0.8)
#    train_data = data.loc[:train_split]
#    test_data = data.loc[train_split:]
    
    msk = np.random.rand(len(data)) < 0.8
    train_data, test_data = data[msk], data[~msk]
    
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", save_top_k=2, monitor="valid_loss", filename="nhits-{epoch:02d}-{val_loss:.2f}")
    models = [
            NHITS(input_size=h, h=h, max_steps=200, start_padding_enabled=True, default_root_dir = "checkpoints", enable_checkpointing=True, callbacks = [checkpoint_callback])
             ]
    model_name = "NHITS"

    if not args.model_path is None:
        model = NHITS.load_from_checkpoint(args.model_path)
        models[0] = model
    
    nf = NeuralForecast(models = models, freq = TIME_INTERVALS[idx])
    
    nf.fit(df=train_data, val_size=int(L * 0.1))

    mae = []
    for horizon in range(0, len(test_data), 2*h):
        
        try:
            test_horizon_insample = pd.DataFrame()
            test_horizon_insample["y"] = np.array(test_data["y"][horizon:horizon+2*h])
            
            test_horizon_insample["ds"] = np.array(test_data["ds"].iloc[horizon:horizon+2*h])
            test_horizon_insample["unique_id"] = np.ones(2*h)

            y_hat_forecast = nf.predict(df=test_horizon_insample)
        
        except ValueError:
            break

        #pred = np.concatenate([np.zeros(h//2), y_hat_forecast["NHITS"]]) 
        #plt.plot(test_horizon_insample["y"])
        #plt.plot(np.arange(h, h*2, 1), y_hat_forecast["NHITS"])
        
        mae.append(np.abs(np.array(test_horizon_insample["y"][h:]) - np.array(y_hat_forecast["NHITS"])).mean())
        #plt.show()
        #plt.cla(); plt.clf();

    print ("START=%s STEP=%s Test set MAE = %.7f" % (os.environ["START"], os.environ["STEP"], np.array(mae).mean()))
    exit()
