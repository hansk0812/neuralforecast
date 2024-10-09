from matplotlib import pyplot as plt

from datasetsforecast.utils import Info
from datasetsforecast.long_horizon import LongHorizon 
from datasetsforecast.long_horizon import ETTh1, ETTh2, ETTm1, ETTm2, \
                                            ECL, Exchange, TrafficL, ILI, Weather

def remove_outliers(data, idx):

    OUTLIER_THRESHOLDS = [
                            [0,0], 
                            [0,0],
                            [0,0],
                            [-6,-4.25],
                            [20,15.3],
                            [0,0],
                            [35,25],
                            [0,0],
                            [-150,-30]
                         ]
    
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
    data["y"].plot()
    plt.show()
    
