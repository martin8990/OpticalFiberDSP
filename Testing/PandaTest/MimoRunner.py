import numpy as np
from qampy import signals, impairments, equalisation, phaserec, helpers
from mimo.mimo import FrequencyDomainBlockwizeMimo
from mimo.error_calculator import CMAErrorCalculator, TrainedLMS
from PlotFunctions.MPLMimoPlots import plot_error, plot_taps, plot_constellation
from EvaluationFunctions.MimoEvaluation import *
from Testing.PandaTest.DirectoryCreator import *

import pandas as pd 
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


movavg_taps = 1000
                    





def try_mimo_setting(df,sig,set,mset,trainingSyms,mimodir):
    N = set['N']
    nmodes = set['nmodes']
    n_training_syms = mset['n_training_syms']
    lb = mset['lb']
    mu = mset['mu']
    nloops = mset['nloops']
    ovsmpl = set['ovsmpl']
    t_conv = N-50000
    t_stop = N-1000
    
    if mset['errorcalc'] == 'lms':
        print('LMS Enabled')
        unit = np.sqrt(2)*0.5
        constellation = [unit + 1j * unit,unit -1j*unit,-unit + 1j * unit,-unit - 1j * unit]
        errorcalc = TrainedLMS(trainingSyms[:,:n_training_syms],constellation,lb)
        sig_with_loops = errorcalc.AddTrainingLoops(sig,ovsmpl,nloops)        
        sig_Martin = sig_with_loops.copy()[:,:N + n_training_syms * nloops]
        mimo = FrequencyDomainBlockwizeMimo(nmodes,lb,ovsmpl,mu,errorcalc)
        sig_Martin[:,:],taps_Martin = mimo.equalize_signal(sig_with_loops)
        
    else: 
        errorcalc = CMAErrorCalculator()
        sig_Martin = sig_with_loops.copy()[:,:N]
        mimo = FrequencyDomainBlockwizeMimo(nmodes,lb,ovsmpl,mu,errorcalc)
        sig_Martin[:,:],taps_Martin = mimo.equalize_signal(sig)
    print(sig_Martin.shape)
    #sig_Martin ,ph = phaserec.viterbiviterbi(sig_Martin, 11)
    err_Martin = []            
    for i_mode in range(nmodes):
            err_Martin_ = calculate_radius_directed_error(sig_Martin[i_mode,0:t_stop],1)
            err_Martin.append(mlab.movavg(abs(err_Martin_),movavg_taps))
    try : 
        ber_martin = calculate_BER(sig_Martin,range(t_conv,t_stop))
    except:
        ber_martin = np.ones(nmodes)
    
    title = "Martin_mu {mu}_lb{lb}"
    plot_constellation(sig_Martin[:,t_conv:t_stop],title,True,mimodir)        
    plot_error(err_Martin,title,True,mimodir)
    plot_taps(taps_Martin,True,mimodir)
    final_error_Martin = calculate_final_error(err_Martin,t_conv,t_stop)
    t_conv_martin = calculate_convergence(err_Martin,final_error_Martin)
    df_martin = UpdateDataFrame(set,mset,ber_martin,t_conv_martin,final_error_Martin,"Martin")
    return df.append(df_martin)

def create_mimos(df,sig,set,trainingSyms,sigdir):
    list_mu = [3e-4,8e-4,1e-3]
    list_lb = [80]
    list_errorcalc = ['lms']
    list_n_training_syms = [10000]
    list_n_loops = [1,2,3]
    for mu in list_mu:
        for lb in list_lb:
            for errorcalc in list_errorcalc:
                for n_training_syms in list_n_training_syms:
                    for nloops in list_n_loops:
                        mset = {
                              'mu' : mu,
                              'lb' : lb,
                              'errorcalc' : errorcalc,
                              'n_training_syms' : n_training_syms,
                              'nloops' : nloops 
                               }
                        mimodir = create_dir_for_dict(sigdir,mset)
                        df = try_mimo_setting(df,sig,set,mset,trainingSyms,mimodir)
    return df