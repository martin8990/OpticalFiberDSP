from qampy import signals, impairments, phaserec
import numpy as np
import mimo.mimo as mimo
import mimo.phaserecovery as phaserec
import matplotlib.pyplot as plt
import pyqt_mimo.mimoplot as bmp
import utility.evaluation as eval
import examples.settings as set
from examples.options import *

def equalize(sig,sequence,mimo_set: set.MimoSettings,u_set: set.UpdateSettings,pr_set: set.PhaseRecoverySettings ,showplots=True):
    lbp = pr_set.lbp
    if pr_set.type == PhaseRec.NONE:
        lbp = 0
    
    ovsmpl = mimo_set.ovsmpl
    nmodes = sig.shape[0]
    N = sequence.shape[1]
    ovconj = 1
    lb = mimo_set.lb
    mu = u_set.mu
    ntraining_syms = u_set.num_trainingsyms
    if mimo_set.widely_linear:
        ovconj = 2
    
        
    trainer = mimo.Trainer(sequence,lb,lbp,ntraining_syms,u_set.phaserec_start)
    for i_train in range(len(u_set.error_calculators) - 2):
        sig = trainer.AddTrainingLoop(sig,ovsmpl)
    block_distr = mimo.BlockDistributer(sig,lb,ovsmpl,ovconj,trainer)
    
    if u_set.update_type == set.MimoUpdaterType.TIMEDOMAIN:
        raise NotImplementedError()
        tap_updater = mimo.TimedomainTapupdater(mu,block_distr)
    else:
        tap_updater = mimo.FrequencyDomainTapUpdater(mu,block_distr)

    # Setup error calculators per section
    errorcalcs = []
    for ecalc_type in u_set.error_calculators:
        if ecalc_type == ECalc.LMS:
           ecalc = mimo.TrainedLMS(block_distr)
        elif ecalc_type == ECalc.SBD:
           ecalc = mimo.TrainedSBD(block_distr)
        elif ecalc_type == ECalc.MRD:
            ecalc = mimo.TrainedMRD(block_distr)
        elif ecalc_type == ECalc.CMA:
            ecalc = mimo.CMAErrorCalculator(block_distr)
        errorcalcs.append(ecalc)
    trainer.set_errorcalcs(errorcalcs)
    trainer.discover_constellation_and_find_symindexes()

    # Equalization
     
    if pr_set.type == PhaseRec.INTERNAL:
        phase_recoverer = mimo.BlindPhaseSearcher(block_distr,trainer,pr_set.num_testangles,search_area = pr_set.search_area)
        sig_eq = mimo.equalize_blockwize(block_distr,tap_updater,phase_recoverer,mimo_set.widely_linear)
    else:
        sig_eq = mimo.equalize_blockwize(block_distr,tap_updater,widely_linear = mimo_set.widely_linear)
        if pr_set.type == PhaseRec.EXTERNAL:
           sig_eq[:,ntraining_syms:] = phaserec.blind_phase_search(sig_eq[:,ntraining_syms:],pr_set.num_testangles,trainer.constellation,lbp)
                
    trainer.calculate_ser_ber(sig_eq)

    if showplots:
        # Vizualization
        taps_martin = tap_updater.retrieve_timedomain_taps()
        err_martin = trainer.retrieve_error()
        sig_sym = trainer.sort_sig_per_sym(sig_eq)
        
        if pr_set.type == PhaseRec.INTERNAL:
            phase = np.asarray(phase_recoverer.phase_collection)
            slips_up = np.asarray(phase_recoverer.slips_up)
            slips_down = np.asarray(phase_recoverer.slips_down)

        all_figs = []
        figs = []

        for i_mode in range(nmodes):
            name = "Mode : " + str(i_mode)  
            row_figs = []
            if i_mode % 2 == 0 and i_mode > 0:
                all_figs.append(figs)
                figs = []
            if False:
                row_figs.append(bmp.ConvergencePlot(err_martin[i_mode],name,ntrainingsyms=ntraining_syms,phase=phase[i_mode],slipups=slips_up[i_mode], slipdowns= slips_down[i_mode]))
            else :
                row_figs.append(bmp.ConvergencePlot(err_martin[i_mode],name,ntrainingsyms=ntraining_syms,nloops = trainer.nloops - 1))
            row_figs.append(bmp.ConstellationPlot(sig_sym[i_mode],N,name))
            row_figs.append(bmp.ErrorSymbolPlot(sig_eq[i_mode],i_mode,trainer,N,name))
  
            if mimo_set.widely_linear:
                row_figs.append(bmp.TapsPlotMerged(taps_martin[:,i_mode,:],N,i_mode,name))
              
            else:
                row_figs.append(bmp.TapsPlot(taps_martin[:,i_mode,0],N,name))
            figs.append(row_figs)
        all_figs.append(figs)
        bmp.plot_interactive_mimo(all_figs,int(sig.shape[1] / 4),int(sig.shape[1] / 4) + 10000, mimo_set.name)


    

