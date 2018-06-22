from qampy import signals, impairments, equalisation, phaserec, helpers
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from mimo.mimo import  BlockwizeMimo, CMAErrorCalculator, TrainedLMS
import matplotlib.pyplot as plt
from PlotFunctions.MPLMimoPlots import *
from PlotFunctions.InteractiveMimoPlot import MimoPlotRequest, plot_interactive_mimo
from Impairments.Impairments import *
from EvaluationFunctions.MimoEvaluation import *
import pickle
from qampy.core.filter import *
## Params
        
N = 5 * 10**4

SNR = 40
lb = 32
mu_Martin = 1e-3
mu_Qampy = 3e-4

phase_noise = 100e3 
Ntaps = 16
t_conv = N-50000
t_stop = N-1000

movavg_taps = 1000
ovsmpl = 1
nmodes = 1
training_loops = 0
ntraining_syms = 10000
## Transmission

sig = signals.SignalQAMGrayCoded(4,N , fb=25e9, nmodes=nmodes)*1.05
print(sig.shape[0])
trainingSyms = sig.copy()[:,:ntraining_syms]
#f,pxx = signal.welch(sig[0])
#plt.plot(f,10*np.log10(pxx))

#if ovsmpl > 1:
#    sig = sig.resample(ovsmpl*sig.fb,beta  =1)
#f,pxx = signal.welch(sig[0])
#plt.plot(f,10*np.log10(pxx))

#sig[0] = rrcos_pulseshaping(sig[0], 1, 0.5, 1, taps=1001)
#sig = sig*2
#f,pxx = signal.welch(sig[0])
#plt.plot(f,10*np.log10(pxx))
#plt.show()

#sig = apply_impulse_response_impairment(sig)    

#sig = impairments.change_snr(sig,SNR)
#sig = impairments.apply_phase_noise(sig,phase_noise)

#matrix = build_mixing_matrix(nmodes,0.4,0.3)
#sig = apply_mltichnl_delayed_matrix_impairment(sig,0,matrix)
#err_Rx = calculate_radius_directed_error(sig[1],1)
#err_Rx = mlab.movavg(abs(err_Rx),movavg_taps)
#plot_request_Rx = MimoPlotRequest(err_Rx,sig.copy()[1],np.zeros(lb*2),"Recieved")

## Equalisation

#taps_QAMPY, err = equalisation.equalise_signal(sig, mu_Qampy, Ntaps=Ntaps, method="cma")
#sig_QAMPY = equalisation.apply_filter(sig, taps_QAMPY)
#sig_QAMPY, ph = phaserec.viterbiviterbi(sig_QAMPY, 11)


unit = np.sqrt(2)*0.5
constellation = [unit + 1j * unit,unit -1j*unit,-unit + 1j * unit,-unit - 1j * unit]
errorcalc = TrainedLMS(trainingSyms,constellation,lb)
sig_with_loops = errorcalc.AddTrainingLoops(sig,ovsmpl,training_loops)
sig_Martin = sig.copy()[:,:N + training_loops *  ntraining_syms]
errorcalc = CMAErrorCalculator()

mimo = BlockwizeMimo(nmodes,True,lb,ovsmpl,mu_Martin,errorcalc)
sig_Martin[:,:],taps_Martin = mimo.equalize_signal(sig_with_loops,True)
sig_Martin ,ph = phaserec.viterbiviterbi(sig_Martin, 11)

err_Martin = []
for i_mode in range(nmodes):
        err_Martin_ = calculate_radius_directed_error(sig_Martin[i_mode,0:t_stop],1)
        err_Martin.append(mlab.movavg(abs(err_Martin_),movavg_taps))

#err_Qampy = mlab.movavg(abs(err[1]),movavg_taps)  

try : 
    print("BER_Martin = ",calculate_BER(sig_Martin,range(t_conv,t_stop)))
except:
    print("BER failed")
#print("BER_Qampy = ", sig_QAMPY.cal_ber())

#plot_constellation(sig,'Origin',False)
#plot_constellation(sig_Martin[:,t_conv:t_stop],'Martin',False)
#plot_taps(taps_Martin[:,:,-1],False)
#plot_taps(taps_QAMPY,False)
#
#plot_error(err_Martin,'Martin',False,'')
#plt.show()


#plot_requests = []
#even_range = range(0,lb*ovsmpl,2)
#odd_range = range(1,lb*ovsmpl + 1,2)

#even_taps = taps_Martin[:,:,:,even_range]
#odd_taps = taps_Martin[:,:,:,odd_range]
#taps = taps_Martin.reshape(ovsmpl,1,taps_Martin.shape[2],lb)
#taps[0,0,:,:] = even_taps
#taps[1,0,:,:] = odd_taps
#plot_taps(even_taps[:,:,-1,:],False)
#plot_taps(odd_taps[:,:,-1,:],False)

#plt.show()
plot_requests = []
for i_mode in range(nmodes):
    plot_requests.append(MimoPlotRequest(err_Martin[i_mode],sig_Martin[i_mode],taps_Martin[:,i_mode],"Mode : " + str(i_mode)))
#plot_request_Qampy = MimoPlotRequest(err_Qampy,sig_QAMPY[1],taps_QAMPY[1,1].reshape(1,Ntaps), "Qampy")
plot_interactive_mimo(plot_requests,t_conv,t_conv + 10000)
