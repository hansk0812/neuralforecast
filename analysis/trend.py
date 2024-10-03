from data import create_uniform_data

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def create_trend_data(data_size, spacing, trend_type="increasing"):

    assert trend_type in ["increasing", "decreasing", "cyclical", "triangle", "triangle_gap"]
    df = create_uniform_data(data_size, spacing)
    
    if trend_type == "increasing":

        # Increasing only trend
        p1, p2 = (0, 0), (data_size, np.random.rand(1)[0] * df["y"].max())
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        c = p1[1] - m * p1[0]
        x = np.arange(0, spacing, spacing / data_size)
        y = ((m*x + c) * data_size / spacing)
        df["y"] = df["y"] * y

        return df
    
    elif trend_type == "decreasing":
        
        # Decreasing only trend
        p1, p2 = (0, df["y"].max()), (spacing, 0) 
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        c = p1[1] - m * p1[0]
        x = np.arange(0, spacing, spacing / data_size)
        y = m*x + c
        df["y"] = df["y"] * y
        
        return df

    elif trend_type == "cyclical":

        # Cyclical Decreasing trend
        p1, p2 = (0, df["y"].max()), (spacing//2, 0) 
        p3, p4 = (spacing//2, df["y"].max()), (spacing, 0) 
        m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
        m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])
        c1 = p1[1] - m1 * p1[0]
        c2 = p3[1] - m1 * p3[0]
        x1 = np.arange(0, spacing//2, spacing / data_size)
        x2 = np.arange(spacing//2, spacing, spacing / data_size)
        y1 = m1*x1 + c1
        y2 = m2*x2 + c2
        df["y"][:data_size//2] = df["y"][:data_size//2] * y1
        df["y"][data_size//2:] = df["y"][data_size//2:] * y2
        
        return df
    
    elif trend_type == "triangle":

        # Triangle
        p1, p2 = (0, 0), (spacing//2, df["y"].max())
        p3, p4 = (spacing//2, df["y"].max()), (spacing, 0) 
        m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
        m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])
        c1 = p1[1] - m1 * p1[0]
        c2 = p3[1] - m2 * p3[0]
        x1 = np.arange(0, spacing//2, spacing / data_size)
        x2 = np.arange(spacing//2, spacing, spacing / data_size)
        y1 = m1*x1 + c1
        y2 = m2*x2 + c2
        df["y"][:data_size//2] = df["y"][:data_size//2] * y1
        df["y"][data_size//2:] = df["y"][data_size//2:] * y2
        
        return df

    elif trend_type == "triangle_gap":

        # Triangle Gap
        p1, p2 = (0, 0), (spacing//4, df["y"].max())
        p3, p4 = ((3*spacing)//4, df["y"].max()), (spacing, 0)
        m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
        m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])
        c1 = p1[1] - m1 * p1[0]
        c2 = p3[1] - m2 * p3[0]
        x1 = np.arange(0, spacing//4, spacing / data_size)
        x2 = np.arange(3*spacing//4, spacing, spacing / data_size)
        y1 = m1*x1 + c1
        y2 = m2*x2 + c2
        df["y"][:data_size//4] = df["y"][:data_size//4] * y1
        df["y"][(3*data_size)//4:data_size] = df["y"][(3*data_size)//4:data_size] * y2
        df["y"][data_size//4:(3*data_size)//4] = 0
        
        return df

if __name__ == "__main__":

    df = create_trend_data(1024, 64, "triangle_gap")
    df["y"].plot()
    plt.show()
    #plt.plot(x, get_decreasing_only_trend(df)["y"])
    #plt.show()
    #plt.plot(x, get_cyclical_decreasing_trend(df)["y"])
    #plt.show()
    #plt.plot(x, get_triangle_trend(df)["y"])
    #plt.show()
    #plt.plot(x, get_gap_trend(df)["y"])
    #plt.show()
