from qampy import signals, impairments, equalisation, phaserec, helpers
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from MIMO.MIMO_CPU import MIMOSettings, fd_cma_mimo_Martin
import matplotlib.pyplot as plt

from PlotFunctions.InteractiveMimoPlot import MimoPlotRequest, plot_interactive_mimo
from EvaluationFunctions.MimoEvaluation import *
## Params

import tkinter as tk
from tkinter import filedialog
import pickle

root = tk.Tk()
root.withdraw()

filename = filedialog.askopenfilename()

file = open(filename,'rb')
sig = pickle.load(file)
N = len(sig[0])
print(N)
mu_Martin = 3e-4
mu_Qampy = 3e-4

lb = 80
R2 = 0.5

t_conv = N-50000
t_stop = N-1000

movavg_taps = 1000
resample = False
  

err_Rx = mlab.movavg(abs(calculate_radius_directed_error(sig[1],R2)),movavg_taps)
plot_request_Rx = MimoPlotRequest(err_Rx,sig.copy()[1],np.zeros(lb*2),"Recieved")


## Equalisation

taps_QAMPY, err = equalisation.equalise_signal(sig, mu_Qampy, Ntaps=61, method="cma")

sig_QAMPY = equalisation.apply_filter(sig, taps_QAMPY)

sig_QAMPY, ph = phaserec.viterbiviterbi(sig_QAMPY, 11)

if resample:
    settings = MIMOSettings(lb = lb,mu = mu_Martin,ovsmpl = 2,R2 = np.sqrt(R2))    
    sig_Martin,taps_Martin = fd_cma_mimo_Martin(sig.copy(),settings)
else:
    settings = MIMOSettings(lb = lb,mu = mu_Martin,ovsmpl = 1,R2 = np.sqrt(R2))
    sig_Martin,taps_Martin = fd_cma_mimo_Martin(sig.copy(),settings)

sig_Martin ,ph = phaserec.viterbiviterbi(sig_Martin, 11)

err_Martin = calculate_radius_directed_error(sig_Martin[1][0:t_stop],np.sqrt(R2))
err_Martin = mlab.movavg(abs(err_Martin),movavg_taps)
err_Qampy = mlab.movavg(abs(err[1]),movavg_taps)  

try : 
    print("BER_Martin = ",calculate_BER(sig_Martin,range(t_conv,t_stop)))
except:
    print("BER failed")
print("BER_Qampy = ", sig_QAMPY.cal_ber())


plot_request_martin = MimoPlotRequest(err_Martin,sig_Martin[1],taps_Martin[1,1,0,:],"Martin")
plot_request_Qampy = MimoPlotRequest(err_Qampy,sig_QAMPY[1],taps_QAMPY[1,1], "Qampy")
plot_interactive_mimo([plot_request_Rx,plot_request_martin,plot_request_Qampy],t_conv,t_conv + 10000)
