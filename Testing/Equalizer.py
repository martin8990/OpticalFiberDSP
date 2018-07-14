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

def equalize(sig,sequence,mu_martin = 1e-3 ,lb = 64,ntraining_syms = 15000,internal_phaserec = True ,widely_linear = True,train_phaserec =False,hybrid = False,equalizer = "AOLMS",showplots = False):
    # 400 max
    lbp = 0
    if internal_phaserec:
        lbp = 10
    ovsmpl = 2
 
    nmodes = sig.shape[0]
    N = sequence.shape[1]
    ovconj = 1
    if widely_linear:
        ovconj = 2
    
    trainer = mimo.Trainer(sequence,lb,lbp,ntraining_syms)
    block_distr = mimo.BlockDistributer(sig,lb,ovsmpl,ovconj,trainer)
    if hybrid:
        tap_updater = mimo.TimedomainTapupdater(mu_martin,block_distr)
    else:
        tap_updater = mimo.FrequencyDomainTapUpdater(mu_martin,block_distr)
    if equalizer == "SBD":
        print("Using SBD")
        errorcalc = mimo.TrainedSBD(block_distr)
    elif equalizer == "AOLMS":
           print("Using Amplitude only LMS")
           errorcalc = mimo.TrainedLMSAmpOnly(block_distr)
    elif equalizer == "MRD":
           print("Using Multi modulus radial directed error calculation")
           errorcalc = mimo.TrainedMRD(block_distr)
    else:
        print("Using LMS")
        errorcalc = mimo.TrainedLMS(block_distr)

    if internal_phaserec:
        phase_recoverer = mimo.BlindPhaseSearcher(block_distr,trainer,40,use_training=train_phaserec)
        sig_martin = mimo.equalize_blockwize(block_distr,tap_updater,errorcalc,phase_recoverer,widely_linear)
    else:
        sig_martin = mimo.equalize_blockwize(block_distr,tap_updater,errorcalc,widely_linear = widely_linear)
        sig_martin[:,ntraining_syms:] = phaserec.blind_phase_search(sig_martin[:,ntraining_syms:],40,trainer.constellation,10)
    


    taps_martin = tap_updater.retrieve_timedomain_taps()
    err_martin = errorcalc.retrieve_error()
    sig_sym = trainer.sort_sig_per_sym(sig_martin)
    
    trainer.calculate_ser_ber(sig_martin)
            
    if internal_phaserec:
        phase = np.asarray(phase_recoverer.phase_collection)
        slips_up = np.asarray(phase_recoverer.slips_up)
        slips_down = np.asarray(phase_recoverer.slips_down)
    if showplots:
        all_figs = []
        figs = []

        for i_mode in range(nmodes):
            name = "Mode : " + str(i_mode)  
            row_figs = []
            if i_mode%2 == 0 and i_mode>0:
                all_figs.append(figs)
                figs = []
            if False:
                row_figs.append(bmp.ConvergencePlot(err_martin[i_mode],name,ntrainingsyms=ntraining_syms,phase=phase[i_mode],slipups=slips_up[i_mode], slipdowns= slips_down[i_mode]))
            else :
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
        bmp.plot_interactive_mimo(all_figs,int(sig.shape[1]/4),int(sig.shape[1]/4)+10000,equalizer)


    

