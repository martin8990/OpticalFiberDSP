from qampy import signals, impairments, equalisation, phaserec, helpers
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from mimo.mimo import FrequencyDomainBlockwizeMimo
from mimo.error_calculator import CMAErrorCalculator, TrainedLMS
import matplotlib.pyplot as plt


from PlotFunctions.MPLMimoPlots import plot_error, plot_taps, plot_constellation
from Impairments.Impairments import *
from EvaluationFunctions.MimoEvaluation import *
from Testing.AutoTest.DirectoryCreator import *
import pandas as pd 
import pickle
## Params
        

def build_and_impair_signal(set, sigdir):
    nmodes= set['nmodes']
    sig = signals.SignalQAMGrayCoded(4,set['N'] , fb=25e9, nmodes=nmodes)
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
    return ovsmpl, sig, sig_Martin, trainingSyms
def UpdateDataFrame(set,mset,ber,samp_converged,final_error,type : str):
    df_temp = pd.DataFrame()
    for i_mode in range(set['nmodes']):
        d = {
            'type' : [type],
            "mode" : [i_mode],
            'ber' : [ber[i_mode]],
            'samp_converged' : [samp_converged[i_mode]],
            'final_error' : [final_error[i_mode]]                
            }      
        df_mode = pd.DataFrame(data= d,index = [0])
        df_set = pd.DataFrame([set],index = [0])
        df_mset = pd.DataFrame([mset],index = [0])
                
        
        df_mode = pd.concat([df_mode,df_set,df_mset],axis = 1)
        #print(df_mode)
        df_temp = df_temp.append(df_mode)
    return df_temp


print("initialized")
testDir = create_dir_for_dsp_evaluation()
movavg_taps = 1000
df = pd.DataFrame()

list_N = [130000]
list_nmodes= [2]
list_ovsmpl = [2]
list_snr = [15]
list_pmd = [1e-9]
list_mixing = [1e-9]
list_delay = [0]
list_loss = [0.3]
list_phase_noise[100e3]
list_impules_impaired = [False]
nreps = 3
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
                                         set = {
                                        'nmodes' : [nmodes],
                                        "ovsmpl" : [ovsmpl],
                                        'snr' : [snr],
                                        'pmd' : [pmd],
                                        'mixing' : [mixing],
                                        'delay' : [delay],
                                        'loss' : [loss],
                                        'phase_noise' : [phase_noise],
                                        'impulse_impaired' : [impulse_impaired]
                                        }
                                         ovsmpl, sig, sig_Martin, trainingSyms = create_dir_for_signalrow(testDir,signal_settings,i_row)
                                         

        
#df = df.sort_values(['mode','pmd','snr','ber'],ascending=[0,0,1,1])
print(df)
df.to_csv(testDir + "\Results.csv", sep=',')


