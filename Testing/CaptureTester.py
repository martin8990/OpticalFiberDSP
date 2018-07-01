import numpy as np
import mimo.mimo as mimo
import matplotlib.pyplot as plt
from PlotFunctions.MPLMimoPlots import *
import EvaluationFunctions.MimoEvaluation as eval
## Params
import scipy.io as sio       
import pyqt_mimo.mimoplot as bmp
import Testing.CaptureLoader as load
N = 7 *10**4

sequence,sig = load.load_3mode_hard_capture_corr()
corrs = []
for i_mode in range(sequence.shape[0]):
    corr = np.correlate((sequence[i_mode]),sig[i_mode,1:sequence.shape[1]*2:2],mode = 'full')
    corrs.append(corr)
 

plt.figure()
plt.title('Sequence -> sig')
for i_mode in range(sequence.shape[0]):
    plt.plot(np.abs(corrs[i_mode]),label = "mode : " + str(i_mode))
    middle = int(len(corrs[i_mode])/2)
    peak = corrs[i_mode].argmax()
    delta = middle - peak
    print("Mode " + str(i_mode) + " has offset " +str( delta)+ " and maxcor is " + str(corrs[i_mode].max()))
plt.legend()
plot_constellation(sig,'test',False)

plt.show()



