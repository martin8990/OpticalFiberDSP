from qampy import signals, impairments, equalisation, phaserec, helpers
import numpy as np
import matplotlib.pyplot as plt

from PlotFunctions.MPLMimoPlots import plot_error, plot_taps, plot_constellation
from Impairments.Impairments import *
from Testing.PandaTest.DirectoryCreator import *
from Testing.PandaTest.MimoRunner import *
import pickle

import pandas as pd 
## Params
        

def build_and_impair_signal(set, sigdir):
    nmodes= set['nmodes']
    N = set['N']
    t_conv = N-50000
    t_stop = N-1000
    sig = signals.SignalQAMGrayCoded(4,N , fb=25e9, nmodes=nmodes)
    sig_Martin = sig.copy()
    trainingSyms = sig.copy()
    ovsmpl = set['ovsmpl']
    if ovsmpl>1:
        sig = sig.resample(ovsmpl*sig.fb,beta=0.1, renormalise=True)    
        
    if set['impulse_impaired']:
            sig = apply_impulse_response_impairment(sig)
    
    matrix = build_mixing_matrix(nmodes,set['mixing'],set['loss'])
    sig = apply_mltichnl_delayed_matrix_impairment(sig,set['delay'],matrix)
    if set['pmd']>0:
        for i_dmode in range(int(nmodes/2)):      
            sig[i_dmode*2:i_dmode*2 + 2] = impairments.apply_PMD(sig[i_dmode*2:i_dmode*2 + 2], np.pi/5.6, set['pmd'])
    sig = impairments.change_snr(sig,set['snr'])
    sig = impairments.apply_phase_noise(sig,set['phase_noise'])
    plot_constellation(sig[:,t_conv:t_stop],"Recieved",True,sigdir)
    return sig, sig_Martin, trainingSyms



def EvaluateSignals(df,testDir):
    list_N = [130000]
    list_nmodes= [4]
    list_ovsmpl = [2]
    list_snr = [15]
    list_pmd = [0]
    list_mixing = [0.4]
    list_delay = [0]
    list_loss = [0.3]
    list_phase_noise= [100e3]
    list_impules_impaired = [False]
    nreps = 1
    for N in list_N:
        for nmodes in list_nmodes:
            for ovsmpl in list_ovsmpl:
                for snr in list_snr:
                    for pmd in list_pmd:
                        for mix in list_mixing:
                            for delay in list_delay:
                                for loss in list_loss:
                                    for phase_noise in list_phase_noise:
                                        for impulse_impaired in list_impules_impaired:
                                            for rep in range(nreps):
                                                
                                                 set = {
                                                     'N' : N,
                                                'nmodes' : nmodes,
                                                "ovsmpl" : ovsmpl,
                                                'snr' : snr,
                                                'pmd' : pmd,
                                                'mixing' : mix,
                                                'delay' : delay,
                                                'loss' : loss,
                                                'phase_noise' : phase_noise,
                                                'impulse_impaired' : impulse_impaired,
                                                'rep' : rep
                                                }                 
                                                 
                                                 sigdir =create_dir_for_dict(testDir,set)
                                                 sig, sig_Martin, trainingSyms = build_and_impair_signal(set,sigdir)
                                                 print(len(sig))
                                                 filename = sigdir + "/_rep" + str(rep) + ".txt"
                                                 file = open(filename,'wb')
                                                 pickle.dump(sig,file)
                                                 filename = sigdir + "/training_rep" + str(rep) + ".txt"
                                                 file = open(filename,'wb')
                                                 pickle.dump(trainingSyms,file)

                                                 df = create_mimos(df,sig,set,trainingSyms,sigdir)
    return df
                                         
