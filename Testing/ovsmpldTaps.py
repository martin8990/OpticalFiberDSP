from qampy import signals, impairments, equalisation, phaserec, helpers
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from mimo.mimo import FrequencyDomainBlockwizeMimo, CMAErrorCalculator
import matplotlib.pyplot as plt
from PlotFunctions.MPLMimoPlots import *
from PlotFunctions.InteractiveMimoPlot import MimoPlotRequest, plot_interactive_mimo
from PlotFunctions.HeatMap import *
from Testing.Impairments import apply_impulse_response_impairment, apply_2chnl_delayed_matrix_impairment
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
ovsmpl = 2

## Transmission
even_samples = range(0,N*2,2)
uneven_samples = range(1,N*2,2)

sig = signals.SignalQAMGrayCoded(4,N , fb=25e9, nmodes=2)
print(sig.shape[0])

if ovsmpl > 1:
    sig = sig.resample(ovsmpl*sig.fb)
#sig = apply_impulse_response_impairment(sig)    
#sig = apply_2chnl_delayed_matrix_impairment(sig,0,int(25e9))
#sig = impairments.apply_PMD(sig, np.pi/5.6, PMD)
#sig = impairments.change_snr(sig,SNR)
#sig = impairments.apply_phase_noise(sig,phase_noise)

err_Rx = calculate_radius_directed_error(sig[1,even_samples],1)
err_Rx = mlab.movavg(abs(err_Rx),movavg_taps)
plot_request_Rx = MimoPlotRequest(err_Rx,sig.copy()[1,even_samples],np.zeros(lb*2),"Recieved")

## Equalisation


sig_Martin = sig.copy()
mimo_evenOnly = FrequencyDomainBlockwizeMimo(2,lb,1,mu_Martin,CMAErrorCalculator())
mimo_unevenOnly = FrequencyDomainBlockwizeMimo(2,lb,1,mu_Martin,CMAErrorCalculator())
mimo = FrequencyDomainBlockwizeMimo(2,lb,ovsmpl,mu_Martin,CMAErrorCalculator())
sig_Martin_evenOnly = sig[:,even_samples]
sig_Martin_evenOnly[:,:],taps_even_only = mimo_evenOnly.equalize_signal(sig_Martin[:,even_samples])

sig_Martin_unevenOnly = sig[:,uneven_samples]
sig_Martin_unevenOnly[:,:],taps_uneven_only = mimo_evenOnly.equalize_signal(sig_Martin[:,uneven_samples])

sig_Martin_regular = sig[:,even_samples]
sig_Martin_regular[:,:],taps_Martin_regular = mimo.equalize_signal(sig_Martin)

err_Martin_regular = calculate_radius_directed_error(sig_Martin_regular[1][0:t_stop],1)
err_Martin_evenOnly = calculate_radius_directed_error(sig_Martin_evenOnly[1][0:t_stop],1)
err_Martin_unevenOnly = calculate_radius_directed_error(sig_Martin_unevenOnly[1][0:t_stop],1)
err_Martin_regular = mlab.movavg(abs(err_Martin_regular),movavg_taps)
err_Martin_evenOnly= mlab.movavg(abs(err_Martin_evenOnly),movavg_taps)
err_Martin_unevenOnly= mlab.movavg(abs(err_Martin_unevenOnly),movavg_taps)


#plot_constellation(sig,'Origin',False)
#plot_constellation(sig_Martin_regular[:,t_conv:t_stop],'Martin',False)
#plot_taps(taps_Martin_regular[:,:,0],False)
#plot_taps(taps_Martin_regular[:,:,1],False)

plot_request_regular = MimoPlotRequest(err_Martin_regular,sig_Martin_regular[1],taps_Martin_regular[1,1,0,:] + taps_Martin_regular[1,1,1,:],"Regular")
plot_request_evenOnly= MimoPlotRequest(err_Martin_evenOnly,sig_Martin_evenOnly[1],taps_even_only[1,1,0,:],"Even only")
plot_request_unevenOnly= MimoPlotRequest(err_Martin_unevenOnly,sig_Martin_unevenOnly[1],taps_uneven_only[1,1,0,:],"Uneven only")

plot_interactive_mimo([plot_request_Rx,plot_request_regular,plot_request_evenOnly,plot_request_unevenOnly],t_conv,t_conv + 10000)

plt.show()
