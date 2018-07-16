from qampy import signals, impairments, equalisation, phaserec, helpers
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from mimo.mimo import FrequencyDomainBlockwizeMimo
from mimo.error_calculator import CMAErrorCalculator, TrainedLMS
import matplotlib.pyplot as plt


from PlotFunctions.MPLMimoPlots import plot_error, plot_taps, plot_constellation
from Testing.Impairments import *
from EvaluationFunctions.MimoEvaluation import *
from Testing.DirectoryCreator import *
import pandas as pd 
import pickle
## Params
        
print("initialized")
testDir = create_dir_for_dsp_evaluation()
movavg_taps = 1000
df = pd.DataFrame()
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

signal_settings = pd.read_csv(os.getcwd() + '\\SignalSettings.csv')
print(signal_settings)

mimo_settings = pd.read_csv(os.getcwd() + '\\MimoSettings.csv')

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

for i_row,set in signal_settings.iterrows():
    N = set['N']
    t_conv = N-50000
    t_stop = N-1000
    nmodes = set['nmodes']
    sigdir = create_dir_for_signalrow(testDir,signal_settings,i_row)
    ovsmpl, sig, sig_Martin, trainingSyms = build_and_impair_signal(set, sigdir)           
    for i_mrow,mset in mimo_settings.iterrows():
        mimodir = create_dir_for_signalrow(sigdir,mimo_settings,i_mrow)
        if mset['errorcalc'] == 'lms':
            print('LMS Enabled')
            unit = np.sqrt(2)*0.5
            constellation = [unit + 1j * unit,unit -1j*unit,-unit + 1j * unit,-unit - 1j * unit]
            errorcalc = TrainedLMS(trainingSyms,constellation,mset['n_training_syms'],mset['lb'])
        else: 
            errorcalc = CMAErrorCalculator()
        mimo = FrequencyDomainBlockwizeMimo(set['nmodes'],mset['lb'],ovsmpl,mset['mu'],errorcalc)

        sig_Martin[:,:],taps_Martin = mimo.equalize_signal(sig)
        sig_Martin ,ph = phaserec.viterbiviterbi(sig_Martin, 11)
        err_Martin = []            
        for i_mode in range(set['nmodes']):
                err_Martin_ = calculate_radius_directed_error(sig_Martin[i_mode,0:t_stop],1)
                err_Martin.append(mlab.movavg(abs(err_Martin_),movavg_taps))
        try : 
            ber_martin = calculate_BER(sig_Martin,range(t_conv,t_stop))
        except:
            ber_martin = np.ones(nmodes)
    
        title = "Martin_mu" + str(mset['mu'])+"_lb"+str(mset['lb'])
        plot_constellation(sig_Martin[:,t_conv:t_stop],title,True,mimodir)        
        plot_error(err_Martin,title,True,mimodir)
        plot_taps(taps_Martin[:,:,0,:],True,mimodir)
        final_error_Martin = calculate_final_error(err_Martin,t_conv,t_stop)
        t_conv_martin = calculate_convergence(err_Martin,final_error_Martin)
        df_martin = UpdateDataFrame(set,mset,ber_martin,t_conv_martin,final_error_Martin,"Martin")
        df = df.append(df_martin)

#df = df.sort_values(['mode','pmd','snr','ber'],ascending=[0,0,1,1])
print(df)
df.to_csv(testDir + "\Results.csv", sep=',')


