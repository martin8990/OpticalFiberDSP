from qampy import signals, impairments, equalisation, phaserec, helpers
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from mimo.mimo import FrequencyDomainBlockwizeMimo
from mimo.error_calculator import TrainedLMS
import matplotlib.pyplot as plt
from PlotFunctions.MPLMimoPlots import *
from PlotFunctions.InteractiveMimoPlot import MimoPlotRequest, plot_interactive_mimo
from PlotFunctions.HeatMap import *
from Testing.Impairments import apply_impulse_response_impairment, apply_2chnl_delayed_matrix_impairment
from EvaluationFunctions.MimoEvaluation import *
## Params
        
N = 13 * 10**4

PMD = 1e-9
SNR = 40
lb = 80
mu_Martin = 3e-4
mu_Qampy = 3e-4

n_training_syms = 10000
phase_noise = 100e3 

t_conv = N-50000
t_stop = N-1000


movavg_taps = 1000

ovsmpl = 1

## Transmission

sig = signals.SignalQAMGrayCoded(4,N , fb=25e9, nmodes=2)
trainingSyms = sig.copy()
if ovsmpl>1:
    sig = sig.resample(2*sig.fb)
#sig = apply_impulse_response_impairment(sig)    
#sig = apply_2chnl_delayed_matrix_impairment(sig,0,int(25e9))
sig = impairments.apply_PMD(sig, np.pi/5.6, PMD)
sig = impairments.change_snr(sig,SNR)
sig = impairments.apply_phase_noise(sig,phase_noise)

err_Rx = calculate_radius_directed_error(sig[0],1)
err_Rx = mlab.movavg(abs(err_Rx),movavg_taps)
plot_request_Rx = MimoPlotRequest(err_Rx,sig.copy()[0],np.zeros(lb*2),"Recieved")

## Equalisation


sig_Martin = sig.copy()
unit = np.sqrt(2)*0.5
constellation = [unit + 1j * unit,unit -1j*unit,-unit + 1j * unit,-unit - 1j * unit]
errorcalc = TrainedLMS(trainingSyms,constellation,n_training_syms,lb)
mimo = FrequencyDomainBlockwizeMimo(2,lb,ovsmpl,mu_Martin,errorcalc)



sig_Martin = sig_Martin[:,:N] 
sig_Martin[:,:],taps_Martin = mimo.equalize_signal(sig)

err_Martin = calculate_radius_directed_error(sig_Martin[0][0:t_stop],1)
err_Martin = mlab.movavg(abs(err_Martin),movavg_taps)

try : 
    print("BER_Martin = ",calculate_BER(sig_Martin,range(t_conv,t_stop)))
except:
    print("BER failed")

#plot_constellation(sig,'Origin',False)
#plot_error([err_Martin],'error',False,"")
#plot_constellation(sig_Martin[:,t_conv:t_stop],'Martin',False)
#plot_taps(taps_Martin[:1,:1,0],False)
#plt.show()


plot_request_martin = MimoPlotRequest(err_Martin,sig_Martin[0],taps_Martin[0,0,0,:],"Martin")

plot_interactive_mimo([plot_request_Rx,plot_request_martin],t_conv,t_conv + 10000)
