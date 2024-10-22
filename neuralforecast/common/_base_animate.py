import os

from matplotlib import pyplot as plt 
import numpy as np 
from matplotlib.animation import FuncAnimation 

# ASSUMPTION: The plots have the same input size and horizon size
def save_animation(gt, pred, fractional_lines, fname=""):
    # gt: One line
    # pred: One line
    # fractional_lines: List of lists of gt and pred lines as per range(START, 1, STEP)

    fig, ax = plt.subplots() 
    ax.plot(np.arange(0, len(gt), 1), gt, alpha=0.4, label="Ground Truth Time Series") 
    ax.plot(np.arange(len(pred), 2*len(pred), 1), pred, alpha=0.4, label="Predicted output")
    
    line1, = ax.plot([], [], alpha=0.4)
    line2, = ax.plot([], [], alpha=0.4)
    
    def animate(i): 
        
        data_fg, data_fp, fraction = fractional_lines[i]
        extra = int(fraction*len(data_fp))
        line1.set_data(np.arange(extra, len(data_fg)+extra+1, 1), np.concatenate((data_fg, data_fp[0:1])))
        line1.set_alpha(0.4)
        line1.set_label("%.2f%% forwarded input signal" % (float(fraction)*100))

        extra = int(fraction*len(data_fg))
        line2.set_data(np.arange(len(data_fp)+extra, 2*len(data_fp)+extra, 1), data_fp)
        line2.set_alpha(0.4)
        line2.set_label("%.2f%% Forwarded Prediction signal" % (float(fraction)*100))

        legend = plt.legend()

        return line1, line2, legend,
 
    anim = FuncAnimation(fig, animate, 
                        frames = len(fractional_lines), 
                        interval = 5000, 
                        blit = True)

    print ("Saving animation to %s!" % os.getcwd())
 
    anim.save('animation_%s.mp4' % fname, writer = 'ffmpeg', fps = 1)
    
    plt.clf()
    plt.close()
