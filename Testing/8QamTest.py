from qampy import signals, impairments, phaserec
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from PlotFunctions.MPLMimoPlots import *
import Testing.CaptureLoader as load     
#from PlotFunctions.InteractiveMimoPlot import MimoPlotRequest, plot_interactive_mimo
import Impairments.Impairments as imp
import EvaluationFunctions.MimoEvaluation as eval
import pickle
from qampy.core.filter import *
import Testing.Equalizer as eq

## Params
        
N = 10 * 10**4

SNR = 20

phase_noise = 70e3 

ovsmpl = 2
nmodes = 2

sequence,sig= load.load_harder_capture()
sequence = sequence[:nmodes,:N]

while sequence.shape[1] < N:
    sequence = np.append(sequence,sequence,axis = 1)
sequence = sequence[:,:N]

sig = signals.SignalQAMGrayCoded(4,N , fb=25e9, nmodes=nmodes)
sig[:,:] = sequence

if ovsmpl > 1:
    sig = sig.resample(ovsmpl*sig.fb,beta  =1)

sig = impairments.change_snr(sig,SNR)
sig = impairments.apply_phase_noise(sig,phase_noise)
matrix = imp.build_mixing_matrix(nmodes,0.2,0.2)
#sig = imp.apply_mltichnl_delayed_matrix_impairment(sig,0,matrix)
eq.equalize_wide(sig,sequence)
#eq.equalize_phaserec(sig,sequence)
#eq.equalize(sig,sequence)

