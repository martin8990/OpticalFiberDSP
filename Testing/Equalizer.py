from qampy import signals, impairments, phaserec
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import mimo.mimo as mimo
import mimo.phaserecovery as phaserec
import matplotlib.pyplot as plt
from PlotFunctions.MPLMimoPlots import *
#from PlotFunctions.InteractiveMimoPlot import MimoPlotRequest, plot_interactive_mimo
import pyqt_mimo.mimoplot as bmp
import EvaluationFunctions.MimoEvaluation as eval

def evaluate_signal(errorcalc, lb, phase_recoverer, sig, tap_updater,constellation,answers,lbp = 0):
    
    N = sig.shape[1]
    nmodes = sig.shape[0]

    taps_martin = tap_updater.retrieve_timedomain_taps()
    err_martin = errorcalc.retrieve_error()
    bit_sigs = eval.seperate_per_bit(constellation,answers,sig,lb,lbp)
    
    offset = int(-lb/2) + lbp
    BER_start = errorcalc.n_training_syms
    BER_stop = N - 10000
    BER = eval.calculate_BER_Martin(sig,answers,constellation,errorcalc.n_training_syms,N-10000,offset)
    print("Calculation Between " ,BER_start," and ", BER_stop)
    print("BER = ",BER)
        
    phase = np.asarray(phase_recoverer.phase_collection)
    slips_up = np.asarray(phase_recoverer.slips_up)
    slips_down = np.asarray(phase_recoverer.slips_down)
    all_figs = []
    figs = []

    for i_mode in range(nmodes):
        name = "Mode : " + str(i_mode)  
        row_figs = []
        if i_mode%2 == 0 and i_mode>0:
            all_figs.append(figs)
            figs = []
        row_figs.append(bmp.ConvergencePlot(err_martin[i_mode],name,ntrainingsyms=errorcalc.n_training_syms,phase = phase[i_mode],slipups=  slips_up[i_mode],slipdowns = slips_down[i_mode]))
        row_figs.append(bmp.ConstellationPlot(bit_sigs[i_mode],N,name))
        if len(taps_martin.shape) == 5:
            if taps_martin.shape[0] < 3:
                row_figs.append(bmp.TapsPlot(taps_martin[:,i_mode,0],N,name))
                row_figs.append(bmp.TapsPlot(taps_martin[:,i_mode,1],N,name + " Conjugated "))
            else:
                row_figs.append(bmp.TapsPlot(taps_martin[i_mode:i_mode+1,i_mode,0],N,name))
                row_figs.append(bmp.TapsPlot(taps_martin[i_mode:i_mode+1,i_mode,1],N,name + " Conjugated "))
        else:
            row_figs.append(bmp.TapsPlot(taps_martin[:,i_mode],N,name))
        figs.append(row_figs)
    all_figs.append(figs)
    bmp.plot_interactive_mimo(all_figs,int(sig.shape[1]/4),int(sig.shape[1]/4)+10000)


def equalize(sig,sequence,phaserec = False ,widely_linear = False ):
   
    constellation,answers = eval.derive_constellation_and_answers(sequence)
    lb = 64
    lbp = 10
    ovsmpl = 2
    ntraining_syms = 15000
    mu_martin = 2e-3
    nmodes = sig.shape[0]
    N = sequence.shape[1]
    ovconj = 1
    if widely_linear:
        ovconj = 2
        block_distr = mimo.BlockDistributer(sig,lb,ovsmpl,ovconj)
        tap_updater = mimo.WideFrequencyDomainTapUpdater(mu_martin,block_distr)
    else :    
        block_distr = mimo.BlockDistributer(sig,lb,ovsmpl,ovconj)
        tap_updater = mimo.FrequencyDomainTapUpdater(mu_martin,block_distr)

    errorcalc = mimo.TrainedLMS(sequence,constellation,block_distr,ntraining_syms,int(-lb/2))

    if phaserec:
        phase_recoverer = mimo.BlindPhaseSearcher(block_distr,sequence,ntraining_syms,40,constellation,lbp)
        sig_martin = mimo.equalize_blockwize(block_distr,tap_updater,errorcalc,phase_recoverer,widely_linear)
    else:
        sig_martin = mimo.equalize_blockwize(block_distr,tap_updater,errorcalc,widely_linear)

    evaluate_signal(errorcalc, lb, phase_recoverer, sig_martin, tap_updater,constellation,answers)


