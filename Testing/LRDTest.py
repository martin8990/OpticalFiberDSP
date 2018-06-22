from qampy import signals, impairments, equalisation, phaserec, helpers
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from mimo.mimo import FrequencyDomainBlockwizeMimo, CMAErrorCalculator, TrainedLMS
import matplotlib.pyplot as plt
from PlotFunctions.MPLMimoPlots import *
from PlotFunctions.InteractiveMimoPlot import MimoPlotRequest, plot_interactive_mimo
from Impairments.Impairments import *
from EvaluationFunctions.MimoEvaluation import *
from mimo.phaserecovery import *
import pickle
## Params
        
N = 6 * 10**4

SNR = 15
lb = 16
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

sig = signals.SignalQAMGrayCoded(4,N , fb=25e9, nmodes=nmodes)
print(sig.shape[0])
trainingSyms = sig.copy()[:,:ntraining_syms]
if ovsmpl > 1:
    sig = sig.resample(ovsmpl*sig.fb)
#sig = apply_impulse_response_impairment(sig)    

sig = impairments.change_snr(sig,SNR)
sig = impairments.apply_phase_noise(sig,phase_noise)

#matrix = build_mixing_matrix(nmodes,0.4,0.3)
#sig = apply_mltichnl_delayed_matrix_impairment(sig,0,matrix)
#err_Rx = calculate_radius_directed_error(sig[1],1)
#err_Rx = mlab.movavg(abs(err_Rx),movavg_taps)
#plot_request_Rx = MimoPlotRequest(err_Rx,sig.copy()[1],np.zeros(lb*2),"Recieved")

## Equalisation



unit = np.sqrt(2)*0.5
constellation = [unit + 1j * unit,unit -1j*unit,-unit + 1j * unit,-unit - 1j * unit]
#errorcalc = TrainedLMS(trainingSyms,constellation,lb)
#sig_with_loops = errorcalc.AddTrainingLoops(sig,ovsmpl,training_loops)
sig_Martin = sig.copy()[:,:N + training_loops *  ntraining_syms]
#errorcalc = CMAErrorCalculator()

#mimo = FrequencyDomainBlockWizeMIMO(nmodes,lb,ovsmpl,mu_Martin,errorcalc)
#sig_Martin[:,:],taps_Martin = mimo.equalize_signal(sig_with_loops,True)
sig_viterbi ,ph = phaserec.viterbiviterbi(sig_Martin, 11)
sig_blind = sig_Martin.copy()



plot_constellation(sig_viterbi,'viterbi',False)
plot_constellation(sig,'Oigin',False)

for test_angle in [30,180]:
    for lbp in [16,32,64,128]:
       sig_blind = blind_phase_search(sig_Martin.copy(),test_angle,constellation,lbp)
       plot_constellation(sig_blind,'Blind, num_angles : ' + str(test_angle) + " lb : " + str(lbp) ,False)
plt.show()



#err_Martin = []
#for i_mode in range(nmodes):
#        err_Martin_ = calculate_radius_directed_error(sig_Martin[i_mode,0:t_stop],1)
#        err_Martin.append(mlab.movavg(abs(err_Martin_),movavg_taps))



#plot_constellation(sig,'Origin',False)
#plot_constellation(sig_viterbi,'Viterbi',False)


##plot_constellation(sig_Martin[:,t_conv:t_stop],'Martin',False)
##plot_taps(taps_Martin[:,:,-1],False)
##plot_taps(taps_QAMPY,False)
##
##plot_error(err_Martin,'Martin',False,'')
#plt.show()


#plot_requests = []

#for i_mode in range(nmodes):
#    plot_requests.append(MimoPlotRequest(err_Martin[i_mode],sig_Martin[i_mode],taps_Martin[:,i_mode],"Mode : " + str(i_mode)))
##plot_request_Qampy = MimoPlotRequest(err_Qampy,sig_QAMPY[1],taps_QAMPY[1,1].reshape(1,Ntaps), "Qampy")
#plot_interactive_mimo(plot_requests,t_conv,t_conv + 10000)
