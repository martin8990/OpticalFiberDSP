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



def equalize_wide_and_phaserec(sig,sequence):
    
    constellation,answers = eval.derive_constellation_and_answers(sequence)
    lb = 64
    lbp = 18
    ovsmpl = 2
    ntraining_syms = 10000
    mu_martin = 1e-3
    nmodes = sig.shape[0]
    N = sequence.shape[1]
    block_distr = mimo.WideBlockDistributer(sig,lb,ovsmpl)
    errorcalc = mimo.TrainedLMS(sequence,constellation,block_distr,ntraining_syms,int(-lb/2) - lbp)

    tap_updater = mimo.WideFrequencyDomainTapUpdater(mu_martin,block_distr)
    phase_recoverer = mimo.BlindPhaseSearcher(block_distr,20,constellation,lbp)

    sig_martin = sequence.copy()
    sig_martin[:,:] = mimo.equalize_blockwize(block_distr,tap_updater,errorcalc,phase_recoverer,True)
    
    taps_martin = tap_updater.retrieve_timedomain_taps()
    err_martin = errorcalc.retrieve_error()
    bit_sigs = eval.seperate_per_bit(constellation,answers,sig_martin,lb,lbp)

    #print("BER_Martin_own_method = ",eval.calculate_BER_Martin(sig_martin,answers,constellation,t_conv,40000,lb,lbp))
        
    phase = np.asarray(phase_recoverer.phase_collection)
    slips_up = np.asarray(phase_recoverer.slips_up)
    slips_down = np.asarray(phase_recoverer.slips_down)

    figs = []
    for i_mode in range(nmodes):
        name = "Mode : " + str(i_mode)  
        row_figs = []
        row_figs.append(bmp.ConvergencePlot(err_martin[i_mode],name,phase = phase[i_mode],slipups=  slips_up[i_mode],slipdowns = slips_down[i_mode]))
        row_figs.append(bmp.ConstellationPlot(bit_sigs[i_mode],N,name))
        #row_figs.append(bmp.TapsPlot(taps_martin[:,i_mode],N,name))
        figs.append(row_figs)
    bmp.plot_interactive_mimo(figs,int(sig.shape[1]/4),int(sig.shape[1]/4)+10000)

def equalize_wide(sig,sequence):
    
    constellation,answers = eval.derive_constellation_and_answers(sequence)
    lb = 64
    lbp = 18
    ovsmpl = 2
    ntraining_syms = 10000
    mu_martin = 1e-3
    nmodes = sig.shape[0]
    N = sequence.shape[1]
    block_distr = mimo.WideBlockDistributer(sig,lb,ovsmpl)
    errorcalc = mimo.TrainedLMS(sequence,constellation,block_distr,ntraining_syms,int(-lb/2) - lbp)

    tap_updater = mimo.WideFrequencyDomainTapUpdater(mu_martin,block_distr)
    phase_recoverer = mimo.BlindPhaseSearcher(block_distr,20,constellation,lbp)

    sig_martin = sequence.copy()
    sig_martin[:,:] = mimo.equalize_blockwize(block_distr,tap_updater,errorcalc,widely_linear=True)
    
    taps_martin = tap_updater.retrieve_timedomain_taps()
    err_martin = errorcalc.retrieve_error()
    bit_sigs = eval.seperate_per_bit(constellation,answers,sig_martin,lb,lbp)

    #print("BER_Martin_own_method = ",eval.calculate_BER_Martin(sig_martin,answers,constellation,t_conv,40000,lb,lbp))
        
    phase = np.asarray(phase_recoverer.phase_collection)
    slips_up = np.asarray(phase_recoverer.slips_up)
    slips_down = np.asarray(phase_recoverer.slips_down)

    figs = []
    for i_mode in range(nmodes):
        name = "Mode : " + str(i_mode)  
        row_figs = []
        row_figs.append(bmp.ConvergencePlot(err_martin[i_mode],name,phase = phase[i_mode],slipups=  slips_up[i_mode],slipdowns = slips_down[i_mode]))
        row_figs.append(bmp.ConstellationPlot(bit_sigs[i_mode],N,name))
        #row_figs.append(bmp.TapsPlot(taps_martin[:,i_mode],N,name))
        figs.append(row_figs)
    bmp.plot_interactive_mimo(figs,int(sig.shape[1]/4),int(sig.shape[1]/4)+10000)


def equalize_phaserec(sig,sequence):
   
    constellation,answers = eval.derive_constellation_and_answers(sequence)
    lb = 64
    lbp = 18
    ovsmpl = 2
    ntraining_syms = 10000
    mu_martin = 1e-3
    nmodes = sig.shape[0]
    N = sequence.shape[1]
    block_distr = mimo.BlockDistributer(sig,lb,ovsmpl)
    errorcalc = mimo.TrainedLMS(sequence,constellation,block_distr,ntraining_syms,int(-lb/2) - lbp)

    tap_updater = mimo.FrequencyDomainTapUpdater(mu_martin,block_distr)
    phase_recoverer = mimo.BlindPhaseSearcher(block_distr,20,constellation,lbp)

    sig_martin = sequence.copy()
    sig_martin[:,:] = mimo.equalize_blockwize(block_distr,tap_updater,errorcalc,phase_recoverer)
    
    taps_martin = tap_updater.retrieve_timedomain_taps()
    err_martin = errorcalc.retrieve_error()
    bit_sigs = eval.seperate_per_bit(constellation,answers,sig_martin,lb,lbp)

    #print("BER_Martin_own_method = ",eval.calculate_BER_Martin(sig_martin,answers,constellation,t_conv,40000,lb,lbp))
        
    phase = np.asarray(phase_recoverer.phase_collection)
    slips_up = np.asarray(phase_recoverer.slips_up)
    slips_down = np.asarray(phase_recoverer.slips_down)

    figs = []
    for i_mode in range(nmodes):
        name = "Mode : " + str(i_mode)  
        row_figs = []
        row_figs.append(bmp.ConvergencePlot(err_martin[i_mode],name,phase = phase[i_mode],slipups=  slips_up[i_mode],slipdowns = slips_down[i_mode]))
        row_figs.append(bmp.ConstellationPlot(bit_sigs[i_mode],N,name))
        #row_figs.append(bmp.TapsPlot(taps_martin[:,i_mode],N,name))
        figs.append(row_figs)
    bmp.plot_interactive_mimo(figs,int(sig.shape[1]/4),int(sig.shape[1]/4)+10000)


def equalize(sig,sequence):
   
    constellation,answers = eval.derive_constellation_and_answers(sequence)
    lb = 64
    lbp = 10
    ovsmpl = 2
    ntraining_syms = 10000
    mu_martin = 1e-3
    nmodes = sig.shape[0]
    N = sequence.shape[1]
    print("YO")
    block_distr = mimo.BlockDistributer(sig,lb,ovsmpl)
    errorcalc = mimo.TrainedLMS(sequence,constellation,block_distr,ntraining_syms,int(-lb/2))

    tap_updater = mimo.FrequencyDomainTapUpdater(mu_martin,block_distr)
    phase_recoverer = mimo.BlindPhaseSearcher(block_distr,40,constellation,lbp)

  
    sig_martin = sequence.copy()
    sig_martin[:,:] = mimo.equalize_blockwize(block_distr,tap_updater,errorcalc)
    
    taps_martin = tap_updater.retrieve_timedomain_taps()
    err_martin = errorcalc.retrieve_error()
    bit_sigs = eval.seperate_per_bit(constellation,answers,sig_martin,lb)

    #print("BER_Martin_own_method = ",eval.calculate_BER_Martin(sig_martin,answers,constellation,t_conv,40000,lb,lbp))
        
    phase = np.asarray(phase_recoverer.phase_collection)
    slips_up = np.asarray(phase_recoverer.slips_up)
    slips_down = np.asarray(phase_recoverer.slips_down)

    figs = []
    for i_mode in range(nmodes):
        name = "Mode : " + str(i_mode)  
        row_figs = []
        row_figs.append(bmp.ConvergencePlot(err_martin[i_mode],name,phase = phase[i_mode],slipups=  slips_up[i_mode],slipdowns = slips_down[i_mode]))
        row_figs.append(bmp.ConstellationPlot(bit_sigs[i_mode],N,name))
        #row_figs.append(bmp.TapsPlot(taps_martin[:,i_mode],N,name))
        figs.append(row_figs)
    bmp.plot_interactive_mimo(figs,int(sig.shape[1]/4),int(sig.shape[1]/4)+10000)
