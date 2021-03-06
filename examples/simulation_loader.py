from qampy import signals, impairments, phaserec
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import utility.impairments as imp
import utility.evaluation as eval
import pickle

from qampy.core.filter import *

def load_8QAM(N):

    SNR = 40

    phase_noise = 0e3 

    ovsmpl = 2
    nmodes = 2

    #sequence,sig= load.load_harder_capture(N)
    #sequence = sequence[:nmodes,:N]

    #while sequence.shape[1] < N:
    #    sequence = np.append(sequence,sequence,axis = 1)
    #sequence = sequence[:,:N]

    sig = signals.SignalQAMGrayCoded(4,N , fb=25e9, nmodes=nmodes)
    #sig[:,:] = sequence
    sequence = sig.copy()
    if ovsmpl > 1:
        sig = sig.resample(ovsmpl*sig.fb,beta  =1)

    sig = impairments.change_snr(sig,SNR)
    sig = impairments.apply_phase_noise(sig,phase_noise)
    matrix = imp.build_mixing_matrix(nmodes,0.2,0.2)
    #sig = imp.apply_mltichnl_delayed_matrix_impairment(sig,0,matrix)
    return sequence,sig
    
