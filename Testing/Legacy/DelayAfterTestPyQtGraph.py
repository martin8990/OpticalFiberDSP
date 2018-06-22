from qampy import signals, impairments, equalisation, phaserec, helpers
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from mimo.mimo import FrequencyDomainBlockwizeMimo, CMAErrorCalculator
import matplotlib.pyplot as plt
from PlotFunctions.MPLMimoPlots import *
from PlotFunctions.InteractiveMimoPlot import MimoPlotRequest, plot_interactive_mimo
from PlotFunctions.HeatMap import *
from Testing.Impairments import apply_impulse_response_impairment, apply_2chnl_delayed_matrix_impairment, apply_2chnl_delayed_after_mix_matrix_impairment
from EvaluationFunctions.MimoEvaluation import *
## Params
        
N = 13 * 10**4

PMD = 1e-9
SNR = 30
lb = 80
mu_Martin = 3e-4
mu_Qampy = 3e-4

phase_noise = 100e3 

t_conv = N-50000
t_stop = N-1000

movavg_taps = 1000
resample = False
matrix = [[1 ,0.2], [0.2, 1]]
## Transmission

sig = signals.SignalQAMGrayCoded(4,N , fb=25e9, nmodes=2)
print(sig.shape[0])
trainingSyms = sig.copy()[0]
#if resample:
#    sig = sig.resample(2*sig.fb)
#sig = apply_impulse_response_impairment(sig)    
sig =  apply_2chnl_delayed_after_mix_matrix_impairment(sig,20,matrix)
#sig = impairments.apply_PMD(sig, np.pi/5.6, PMD)
#sig = impairments.change_snr(sig,SNR)
#sig = impairments.apply_phase_noise(sig,phase_noise)

err_Rx = calculate_radius_directed_error(sig[1],1)
err_Rx = mlab.movavg(abs(err_Rx),movavg_taps)
plot_request_Rx = MimoPlotRequest(err_Rx,sig.copy()[1],np.zeros(lb*2),"Recieved")

## Equalisation

taps_QAMPY, err = equalisation.equalise_signal(sig, mu_Qampy, Ntaps=61, method="cma")
sig_QAMPY = equalisation.apply_filter(sig, taps_QAMPY)
sig_QAMPY, ph = phaserec.viterbiviterbi(sig_QAMPY, 11)

#if resample:
#    sig_Martin = sig[:,:]
#    even_samples = range(0,N*2,2)
#    sig_Martin = sig_Martin[:,even_samples]
#    sig_Martin,taps_Martin = mimo_cma_fd_ba(sig_Martin)
#else:
sig_Martin = sig.copy()
mimo = FrequencyDomainBlockwizeMimo(2,lb,1,mu_Martin,CMAErrorCalculator())
sig_Martin[:,:],taps_Martin = mimo.equalize_signal(sig)

#sig_Martin ,ph = phaserec.viterbiviterbi(sig_Martin, 11)

err_Martin = calculate_radius_directed_error(sig_Martin[0][0:t_stop],1)
err_Martin = mlab.movavg(abs(err_Martin),movavg_taps)
err_Qampy = mlab.movavg(abs(err[1]),movavg_taps)  

try : 
    print("BER_Martin = ",calculate_BER(sig_Martin,range(t_conv,t_stop)))
except:
    print("BER failed")
print("BER_Qampy = ", sig_QAMPY.cal_ber())

plot_constellation(sig,'Origin',False)
plot_constellation(sig_Martin[:,t_conv:t_stop],'Martin',False)
plot_taps(taps_Martin[:,:,0],False)
plt.show()


#
plot_request_martin = MimoPlotRequest(err_Martin,sig_Martin[1],taps_Martin[1,1,0,:],"Martin")
plot_request_Qampy = MimoPlotRequest(err_Qampy,sig_QAMPY[1],taps_QAMPY[1,1], "Qampy")
plot_interactive_mimo([plot_request_Rx,plot_request_martin,plot_request_Qampy],t_conv,t_conv + 10000)

  
