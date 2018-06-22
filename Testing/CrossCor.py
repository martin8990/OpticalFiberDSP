import numpy as np
import mimo.mimo as mimo
import matplotlib.pyplot as plt
from PlotFunctions.MPLMimoPlots import *
import EvaluationFunctions.MimoEvaluation as eval
## Params
import scipy.io as sio       
import pyqt_mimo.mimoplot as bmp
N = 20 *10**4
capture = sio.loadmat('QSM_3.10E-6_51taps.mat', squeeze_me=True)
    
sequence = capture['Sequence'][()].astype(np.complex128)
sequence = np.roll(sequence,24676)
#sequence[1] = np.roll(sequence[1],-23000)
#sequence[1] = np.roll(sequence[1],-23045)


sig = capture['Input'][()].astype(np.complex128)
even = range(0,N*2,2)
corr = np.correlate(sequence[1],sig[0,1:sequence.shape[1]*2:2],mode = 'full')

plt.figure()
plt.title('Sequence -> sig')
plt.plot(np.abs(corr))
plt.show()