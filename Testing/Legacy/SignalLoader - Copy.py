from qampy import signals, impairments, equalisation, phaserec, helpers
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from mimo.mimo import FrequencyDomainBlockwizeMimo, CMAErrorCalculator, TrainedLMS
import matplotlib.pyplot as plt
from PlotFunctions.MPLMimoPlots import *
from PlotFunctions.InteractiveMimoPlot import MimoPlotRequest, plot_interactive_mimo
from PlotFunctions.HeatMap import *
from Impairments.Impairments import apply_impulse_response_impairment, apply_2chnl_delayed_matrix_impairment
from EvaluationFunctions.MimoEvaluation import *
import pickle
## Params
        

import tkinter as tk
from tkinter import filedialog
import pickle

root = tk.Tk()
root.withdraw()

filename = filedialog.askopenfilename()

file = open(filename,'rb')
sig = pickle.load(file)
filename = filedialog.askopenfilename()

file = open(filename,'rb')

trainingSyms = pickle.load(file)

mu_Martin = 3e-4
mu_Qampy = 3e-4



movavg_taps = 1000
ovsmpl = 2
N = len(trainingSyms[0])
nmodes = 2
n_training_syms = 80000
lb = 80
t_conv = N-50000
t_stop = N-1000
## Transmission

err_Rx = calculate_radius_directed_error(sig[1],1)
err_Rx = mlab.movavg(abs(err_Rx),movavg_taps)
plot_request_Rx = MimoPlotRequest(err_Rx,sig.copy()[1],np.zeros(lb*2),"Recieved")

# Equalisation
Ntaps = 61
taps_QAMPY, err = equalisation.equalise_signal(sig, mu_Qampy, Ntaps=Ntaps, method="cma")
sig_QAMPY = equalisation.apply_filter(sig, taps_QAMPY)
sig_QAMPY, ph = phaserec.viterbiviterbi(sig_QAMPY, 11)

sig_Martin = sig.copy()[:,:N]
unit = np.sqrt(2)*0.5
constellation = [unit + 1j * unit,unit -1j*unit,-unit + 1j * unit,-unit - 1j * unit]
errorcalc = TrainedLMS(trainingSyms,constellation,n_training_syms,lb)
mimo = FrequencyDomainBlockwizeMimo(nmodes,lb,ovsmpl,mu_Martin,errorcalc)

sig_Martin[:,:],taps_Martin = mimo.equalize_signal(sig,True)

sig_Martin ,ph = phaserec.viterbiviterbi(sig_Martin, 11)
err_Martin = calculate_radius_directed_error(sig_Martin[0][0:t_stop],1)
err_Martin = mlab.movavg(abs(err_Martin),movavg_taps)
err_Qampy = mlab.movavg(abs(err[1]),movavg_taps)  

try : 
    print("BER_Martin = ",calculate_BER(sig_Martin,range(t_conv,t_stop)))
except:
    print("BER failed")
print("BER_Qampy = ", sig_QAMPY.cal_ber())

#plot_constellation(sig,'Origin',False)
#plot_constellation(sig_Martin[:,t_conv:t_stop],'Martin',False)
#plot_taps(taps_Martin[:,:,0],False)
#plt.show()


#


plot_request_martin = MimoPlotRequest(err_Martin,sig_Martin[1],taps_Martin[1,1,0],"Martin")
plot_request_Qampy = MimoPlotRequest(err_Qampy,sig_QAMPY[1],taps_QAMPY[1,1].reshape(1,Ntaps), "Qampy")
plot_interactive_mimo([plot_request_martin,plot_request_Qampy],t_conv,t_conv + 10000)
