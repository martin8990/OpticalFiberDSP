import numpy as np
import mimo.mimo as mimo
import matplotlib.pyplot as plt
from PlotFunctions.MPLMimoPlots import *
import EvaluationFunctions.MimoEvaluation as eval
## Params
import scipy.io as sio       
import pyqt_mimo.mimoplot as bmp
N = 20 *10**4
capture = sio.loadmat('QSM_3.10E-6_51taps.mat', squeeze_me=True)
    
sequence = capture['Sequence'][()].astype(np.complex128)
sequence[0] = np.roll(sequence[0],24600)
sequence[1] = np.roll(sequence[1],1600)



#sequence = np.roll(sequence,1,axis=0)24676
#sequence[1] = sequence[0]


sig = capture['Input'][()].astype(np.complex128)

#sig = np.roll(sig,1,axis = 0)
#corr = np.correlate(sequence[0],sequence[1],mode='full')

#plt.figure()
#plt.title('Sequence[0] -> Sequence[1]')
#plt.plot(corr)
#plt.show()
# 13370
even = range(0,N*2,2)
corr = np.correlate(sequence[1],sig[1,1:sequence.shape[1]*2:2],mode = 'full')

#plt.figure()
#plt.title('Sequence -> sig')
#plt.plot(np.abs(corr))
#plt.show()


sig = sig*20
sig = sig[:,:N*2]
plot_constellation(sig,"Origin",False)
#plt.figure()
#plt.scatter(sequence[0].real,sequence[0].imag)
#plt.show()

#constel_arr = np.asarray(constellation) 
#plt.scatter(constel_arr.real,constel_arr.imag)
#plt.show()
lb = 256 # multiplied with ovsmpl
mu_martin = 5e-4

t_conv = N-50000
t_stop = N-1000

movavg_taps = 1000
ovsmpl = 2
nmodes = sig.shape[0]
training_loops = 0
ntraining_syms = sequence.shape[1]-10000
for k in range(3):
    sequence = np.concatenate((sequence,sequence),axis = 1)
constellation,answers = eval.derive_constellation_and_answers(sequence)
block_distr = mimo.BlockDistributer(sig,lb,ovsmpl)
errorcalc = mimo.TrainedLMS(sequence,constellation,block_distr,ntraining_syms,-int(lb/2))

#sig_with_loops = errorcalc.AddTrainingLoops(sig,ovsmpl,training_loops)
#sig_martin = sig.copy()[:,:N + training_loops *  ntraining_syms]
 
tap_updater = mimo.FrequencyDomainTapUpdater(mu_martin,block_distr)
phase_recoverer = mimo.BlindPhaseSearcher(block_distr,30,constellation)

sig_martin =  mimo.equalize_blockwize(block_distr,tap_updater,errorcalc)
#sig_martin = mimo.equalize_blockwize_with_phaserec(block_distr,tap_updater,errorcalc,phase_recoverer)
taps_martin = tap_updater.retrieve_timedomain_taps()
err_martin = errorcalc.retrieve_error()

phase = np.asarray(phase_recoverer.phase_collection)
slips_up = np.asarray(phase_recoverer.slips_up)
slips_down = np.asarray(phase_recoverer.slips_down)

bit_sigs = eval.seperate_per_bit(constellation,answers,sig_martin,lb)

figs = []
for i_mode in range(nmodes):
    name = "Mode : " + str(i_mode)  
    row_figs = []
    row_figs.append(bmp.ConvergencePlotBasic(err_martin[i_mode],name))
    row_figs.append(bmp.ConstellationPlot(bit_sigs[i_mode],N,name))
    row_figs.append(bmp.TapsPlot(taps_martin[:,i_mode],N,name))
    figs.append(row_figs)
bmp.plot_interactive_mimo(figs,t_conv,t_conv+10000)