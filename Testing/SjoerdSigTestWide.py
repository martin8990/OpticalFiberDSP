import numpy as np
import mimo.mimo as mimo
import matplotlib.pyplot as plt
from PlotFunctions.MPLMimoPlots import *
import EvaluationFunctions.MimoEvaluation as eval
## Params
import Testing.CaptureLoader as load       
import pyqt_mimo.mimoplot as bmp
N = 10 *10**4

sequence,sig= load.load_harder_capture()
sig = sig[:,:N*2]
#plot_constellation(sig,"Origin",False)
#plt.show()

#plt.scatter(sequence[0].real,sequence[0].imag)
#plt.show()

lb = 64 # multiplied with ovsmpl
mu_martin = 1e-3

t_conv = N-50000
t_stop = N-1000

movavg_taps = 1000
ovsmpl = 2
nmodes = sig.shape[0]
training_loops = 3
ntraining_syms = 5000
for k in range(3):
    sequence = np.concatenate((sequence,sequence),axis = 1)

sequence,sig= eval.AddTrainingLoops(sig,sequence,ovsmpl,training_loops,ntraining_syms)
N = N + training_loops * ntraining_syms
constellation,answers = eval.derive_constellation_and_answers(sequence)
block_distr = mimo.WideBlockDistributer(sig,lb,ovsmpl)
errorcalc = mimo.TrainedLMS(sequence,constellation,block_distr,ntraining_syms + training_loops * ntraining_syms,-int(lb/2))


#sig_with_loops = errorcalc.AddTrainingLoops(sig,ovsmpl,training_loops)
#sig_martin = sig.copy()[:,:N + training_loops *  ntraining_syms]
 
tap_updater = mimo.WideFrequencyDomainTapUpdater(mu_martin,block_distr)
phase_recoverer = mimo.BlindPhaseSearcher(block_distr,30,constellation,6)

sig_martin =  mimo.equalize_blockwize_widely(block_distr,tap_updater,errorcalc)
#sig_martin = mimo.equalize_blockwize_with_phaserec(block_distr,tap_updater,errorcalc,phase_recoverer)
taps_martin = tap_updater.retrieve_timedomain_taps()
err_martin = errorcalc.retrieve_error()

phase = np.asarray(phase_recoverer.phase_collection)
slips_up = np.asarray(phase_recoverer.slips_up)
slips_down = np.asarray(phase_recoverer.slips_down)

bit_sigs = eval.seperate_per_bit(constellation,answers,sig_martin,lb)

#BER = eval.calculate_BER_Martin(sig,answers,constellation,10000,10000,lb,0)
#print("Ber : " + str(BER))

figs = []
for i_mode in range(2):
    name = "Mode : " + str(i_mode)  
    row_figs = []
    row_figs.append(bmp.ConvergencePlot(err_martin[i_mode],name,ntrainingsyms=ntraining_syms,nloops=training_loops))
    row_figs.append(bmp.ConstellationPlot(bit_sigs[i_mode],N,name))
    row_figs.append(bmp.TapsPlot(np.append(taps_martin[:,i_mode,0],taps_martin[:,i_mode,1],axis = 0),N,name))
    figs.append(row_figs)
bmp.plot_interactive_mimo(figs,t_conv,t_conv+10000)