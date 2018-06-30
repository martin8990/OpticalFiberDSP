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





def equalize(sig,sequence,phaserec = False ,widely_linear = False ):
   
    lb = 64
    lbp = 0
    if phaserec:
        lbp = 10
    ovsmpl = 2
    ntraining_syms = 15000
    mu_martin = 2e-3
    nmodes = sig.shape[0]
    N = sequence.shape[1]
    ovconj = 1
    if widely_linear:
        ovconj = 2
   
    trainer = mimo.Trainer(sequence,lb,lbp,ntraining_syms)
    block_distr = mimo.BlockDistributer(sig,lb,ovsmpl,ovconj,trainer)
    tap_updater = mimo.FrequencyDomainTapUpdater(mu_martin,block_distr)
    errorcalc = mimo.TrainedLMS(block_distr)

    if phaserec:
        phase_recoverer = mimo.BlindPhaseSearcher(block_distr,trainer,40)
        sig_martin = mimo.equalize_blockwize(block_distr,tap_updater,errorcalc,phase_recoverer,widely_linear)
    else:
        sig_martin = mimo.equalize_blockwize(block_distr,tap_updater,errorcalc,widely_linear = widely_linear)
    
    taps_martin = tap_updater.retrieve_timedomain_taps()
    err_martin = errorcalc.retrieve_error()
    sig_sym = trainer.sort_sig_per_sym(sig_martin)
    trainer.calculate_SER(sig_martin)
            

    #phase = np.asarray(phase_recoverer.phase_collection)
    #slips_up = np.asarray(phase_recoverer.slips_up)
    #slips_down = np.asarray(phase_recoverer.slips_down)
    all_figs = []
    figs = []

    for i_mode in range(nmodes):
        name = "Mode : " + str(i_mode)  
        row_figs = []
        if i_mode%2 == 0 and i_mode>0:
            all_figs.append(figs)
            figs = []
        row_figs.append(bmp.ConvergencePlot(err_martin[i_mode],name,ntrainingsyms=ntraining_syms))
        row_figs.append(bmp.ConstellationPlot(sig_sym[i_mode],N,name))
        if widely_linear:
            if taps_martin.shape[0] < 3:
                row_figs.append(bmp.TapsPlot(taps_martin[:,i_mode,0],N,name))
                row_figs.append(bmp.TapsPlot(taps_martin[:,i_mode,1],N,name + " Conjugated "))
            else:
                row_figs.append(bmp.TapsPlot(taps_martin[i_mode:i_mode+1,i_mode,0],N,name))
                row_figs.append(bmp.TapsPlot(taps_martin[i_mode:i_mode+1,i_mode,1],N,name + " Conjugated "))
        else:
            row_figs.append(bmp.TapsPlot(taps_martin[:,i_mode,0],N,name))
        figs.append(row_figs)
    all_figs.append(figs)
    bmp.plot_interactive_mimo(all_figs,int(sig.shape[1]/4),int(sig.shape[1]/4)+10000)


    

