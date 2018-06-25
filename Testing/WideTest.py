from qampy import signals, impairments, phaserec
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import mimo.mimo as mimo
import matplotlib.pyplot as plt
from PlotFunctions.MPLMimoPlots import *
#from PlotFunctions.InteractiveMimoPlot import MimoPlotRequest, plot_interactive_mimo
import pyqt_mimo.mimoplot as bmp
import Impairments.Impairments as imp
import EvaluationFunctions.MimoEvaluation as eval
import pickle
from qampy.core.filter import *


## Params
        
N = 5 * 10**4

SNR = 20
lb = 64 
mu_martin = 1e-3

phase_noise = 100e3 
t_conv = N-50000
t_stop = N-1000

ovsmpl = 2
nmodes = 1
training_loops = 0
ntraining_syms = 10000
lbp = 10
## Transmission

sig = signals.SignalQAMGrayCoded(4,N , fb=25e9, nmodes=nmodes)

print(sig.shape[0])
sequence = sig.copy()


if ovsmpl > 1:
    sig = sig.resample(ovsmpl*sig.fb,beta  =1)
    #sig[0] = rrcos_pulseshaping(sig[0], 1, 0.5, 1, taps=1001)

sig = impairments.change_snr(sig,SNR)
#sig = impairments.apply_phase_noise(sig,phase_noise)
constellation,answers = eval.derive_constellation_and_answers(sequence)

#matrix = imp.build_mixing_matrix(nmodes,0.4,0.3)
#sig = imp.apply_mltichnl_delayed_matrix_impairment(sig,0,matrix)


#Equalisation

block_distr = mimo.WideBlockDistributer(sig,lb,ovsmpl)

errorcalc = mimo.TrainedLMS(sequence,constellation,block_distr,ntraining_syms,int(-lb/2))

#sig = errorcalc.AddTrainingLoops(sig,ovsmpl,training_loops)

#errorcalc = mimo.CMAErrorCalculator(block_distr)
tap_updater = mimo.WideFrequencyDomainTapUpdater(mu_martin,block_distr)
phase_recoverer = mimo.BlindPhaseSearcher(block_distr,20,constellation,lbp)

sig_martin = sequence.copy()
sig_martin[:,:] =  mimo.equalize_blockwize_widely(block_distr,tap_updater,errorcalc)
#sig_martin[:,:] = mimo.equalize_blockwize_with_phaserec(block_distr,tap_updater,errorcalc,phase_recoverer)
#print(offset - N/2)

taps_martin = tap_updater.retrieve_timedomain_taps()
err_martin = errorcalc.retrieve_error()
bit_sigs = eval.seperate_per_bit(constellation,answers,sig_martin,lb,0)

#plot_constellation_bitsep(bit_sigs,constellation,"Martin",False)
#plt.show()
#sig_martin = phaserec.blind_phase_search(sig_martin,30,constellation,100)

try : 
    print("BER_Martin = ",eval.calculate_BER(sig_martin,range(t_conv,t_conv+ 5000)))
except:
    print("BER failed")
#print("BER_Qampy = ", sig_QAMPY.cal_ber())

#plot_constellation(sig_martin[:,t_conv:t_stop],'Martin',False)
#plot_taps(taps_martin[:,:,-1],False)
#plot_error(err_martin,'Martin',False,'')
##plt.show()
 
phase = np.asarray(phase_recoverer.phase_collection)
slips_up = np.asarray(phase_recoverer.slips_up)
slips_down = np.asarray(phase_recoverer.slips_down)

figs = []


for i_mode in range(nmodes):
    name = "Mode : " + str(i_mode)  
    row_figs = []
    row_figs.append(bmp.ConvergencePlot(err_martin[i_mode],name,ntrainingsyms=ntraining_syms,nloops=training_loops))
    row_figs.append(bmp.ConstellationPlot(bit_sigs[i_mode],N,name))
    row_figs.append(bmp.TapsPlot(np.append(taps_martin[:,i_mode,0],taps_martin[:,i_mode,1],axis = 0),N,name))
    figs.append(row_figs)
bmp.plot_interactive_mimo(figs,int(sig.shape[1]/4),int(sig.shape[1]/4)+10000)
