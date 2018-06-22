from qampy import signals, impairments, equalisation, phaserec, helpers
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from MIMO.MIMO_CPU import MIMOSettings, FD_mimo_Martin
import matplotlib.pyplot as plt

from PlotFunctions.InteractiveMimoPlot import MimoPlotRequest, plot_interactive_mimo
from PlotFunctions.HeatMap import *
from Tests.Impulse_response_debug import apply_impulse_response_impairment, apply_impulse_response_impairment_FD
from EvaluationFunctions.MimoEvaluation import *
## Params
        
N = 15 * 10**4
PMD = 1e-9
SNR = 15
mu = 5e-4

list_lb = [32,64, 128, 256]
R2 = 0.5
phase_noise = 100e3 

t_conv = N-60000
t_stop = N-1000

movavg_taps = 1000

## Transmission

sig = signals.SignalQAMGrayCoded(4,N , fb=25e9, nmodes=2)

 
sig = impairments.change_snr(sig,SNR)
sig = impairments.apply_PMD(sig, np.pi/5.6, PMD)
sig = impairments.apply_phase_noise(sig,phase_noise)


list_plotRequest = []
## Equalisation
for lb in list_lb:
    sig_rx = sig.copy()
    settings = MIMOSettings(lb,mu,1,R2 = np.sqrt(R2))
    sig_Martin,taps_Martin = FD_mimo_Martin(sig_rx,settings)    
    
    sig_Martin ,ph = phaserec.viterbiviterbi(sig_Martin, 11)
    
    err_Martin = calculate_radius_directed_error(sig_Martin[1][0:t_stop],R2)
    err_Martin = mlab.movavg(abs(err_Martin),movavg_taps)
    list_plotRequest.append(MimoPlotRequest(err_Martin,sig_Martin[1],taps_Martin[1,1,0,:],impulse_Martin,"lb : " + str(lb)))
    try : 
        print("BER " ,lb," = ",calculate_BER(sig_Martin,range(t_conv,t_stop)))
    except:
        print("BER failed due to instability")


#create_heatmap_data_from_constellation(sig_QAMPY[1,20000:40000],200)
#create_heatmap_data_from_constellation(sig_Martin[1,20000:40000],200)

plot_interactive_mimo(list_plotRequest,t_conv,t_conv + 10000)
#plot_interactive_mimo(list_plotRequest[2:4],t_conv,t_conv + 10000)
# -*- coding: utf-8 -*-

