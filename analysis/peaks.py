from scipy.signal import find_peaks
from trend import df_increasing, df_decreasing, df_increasing_sq
from trend import df_cycle, df_triangle, df_gap

from matplotlib import pyplot as plt

def peaks(series):
    
    peaks = find_peaks(series)
    peaks_y = series[peaks[0]]

    plt.plot(peaks[0], peaks_y)
    plt.show()

peaks(df_increasing["y"])
peaks(df_decreasing["y"])
peaks(df_increasing_sq["y"])

peaks(df_cycle["y"])
peaks(df_triangle["y"])
peaks(df_gap["y"])

