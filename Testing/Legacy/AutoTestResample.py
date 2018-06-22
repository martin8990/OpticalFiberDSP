from qampy import signals, impairments, equalisation, phaserec, helpers
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from MIMO.MIMO_CPU import mimo_fd_ba
import matplotlib.pyplot as plt


from PlotFunctions.MPLMimoPlots import plot_error, plot_taps, plot_constellation
from Testing.Impairments import apply_impulse_response_impairment, apply_2chnl_delayed_matrix_impairment
from EvaluationFunctions.MimoEvaluation import *
from Testing.DirectoryCreator import *
import pandas as pd 
import pickle
## Params
        
N = 13 * 10**4

list_pmd =  [1e-9]
list_snr = [17]
list_mu_m = [2e-4,3e-4]
list_mu_q = [1e-4,2e-4]
nrepetitions = 3

list_lb = [64,80]
list_ntaps = [61,81]
R2 = 1
phase_noise = 100e3 

t_conv = N-50000
t_stop = N-1000

movavg_taps = 1000
resample = False
apply_impulse = False
produce_plots = True
apply_matrix = False
print("initialized")
nsignals = len(list_pmd) * len(list_snr)
nqampys = len(list_mu_q) * len(list_ntaps)

testDir = create_dir_for_dsp_evaluation()

df = pd.DataFrame()
def UpdateDataFrame(pmd,snr,mu,ntaps,ber,samp_converged,final_error,repetition,type : str):
    df_temp = pd.DataFrame()
    for i_mode in range(2):
        d = {
            'type' : [type],
            "MODE" : [i_mode],
            "Repetition" : [repetition],
            'PMD' : [pmd],
            'SNR' : [snr],
            'mu' : [mu],
            'ntaps' : [ntaps],
            'BER' : [ber[i_mode]],
            'samp_converged' : [samp_converged[i_mode]],
            'final_error' : [final_error[i_mode]]                
            }      
        df_temp = df_temp.append(pd.DataFrame(data= d))
    return df_temp


for pmd in list_pmd:
    for snr in list_snr:
        sigdir = create_dir_for_signal(testDir,pmd,snr,apply_impulse,resample)
        for r in range(nrepetitions):
            sig = signals.SignalQAMGrayCoded(4,N , fb=25e9, nmodes=2)
    
            if resample:
                sig = sig.resample(2*sig.fb,beta=0.1, renormalise=True)    
        
            if apply_impulse:
                 sig = apply_impulse_response_impairment(sig)
            if apply_matrix:
                sig[:,:] = apply_2chnl_delayed_matrix_impairment(sig,1e-9,25e9)
            sig = impairments.apply_PMD(sig, np.pi/5.6, pmd)
            sig = impairments.change_snr(sig,snr)
            sig = impairments.apply_phase_noise(sig,phase_noise)
     
            plot_constellation(sig[:,t_conv:t_stop],"Recieved",True,sigdir)           

            filename = sigdir + "/_rep" + str(r) + ".txt"
            file = open(filename,'wb')
            pickle.dump(sig,file)
            for mu_q in list_mu_q:
                for ntaps in list_ntaps:
                    mimodir = create_dir_for_mimo_result(sigdir,mu_q,ntaps,"Qampy")
                    taps_qampy, err = equalisation.equalise_signal(sig, mu_q, Ntaps=ntaps, method="cma")
                    sig_qampy = equalisation.apply_filter(sig, taps_qampy)
                    sig_qampy, ph = phaserec.viterbiviterbi(sig_qampy, 11)
            
                    err_qampy = [] 
                    for i_mode in range(2):
                        err_qampy.append( mlab.movavg(abs(err[i_mode]),movavg_taps))
            
                    try:
                        ber_qampy = calculate_BER(sig_qampy,range(t_conv,t_stop))
                    except :
                        ber_qampy = [1,1]
                    if produce_plots:
                        title = "Qampy_mu" + str(mu_q)+"_taps"+str(ntaps)
                        plot_constellation(sig_qampy[:,t_conv:t_stop],title,True,directory= mimodir)        
                        plot_error(err_qampy,title,True,mimodir)
                        plot_taps(taps_qampy,True,mimodir)
                    final_error_Qampy = calculate_final_error(err_qampy,t_conv,t_stop)
                    t_conv_qampy = calculate_convergence(err_qampy,final_error_Qampy)
                    df_qampy = UpdateDataFrame(pmd,snr,mu_q,ntaps,ber_qampy,t_conv_qampy,final_error_Qampy,r,"Qampy")
                    df = df.append(df_qampy)    


            for mu_m in list_mu_m:
                 for lb in list_lb:
                    mimodir = create_dir_for_mimo_result(sigdir,mu_m,lb,"Martin")
                    if resample:
                        sig_Martin = sig.copy()
                        sig_Martin = sig_Martin[:,range(0,N*2,2)]
                        sig_Martin[:,:],taps_Martin = mimo_fd_ba(sig_Martin,ovsmpl= 1,lb = lb,mu = mu_m)
                    else:
                        sig_Martin = sig.copy()
                        sig_Martin[:,:],taps_Martin = mimo_fd_ba(sig_Martin,ovsmpl= 1,lb = lb,mu = mu_m)
                    sig_Martin ,ph = phaserec.viterbiviterbi(sig_Martin, 11)
                    err_Martin = []            
                    for i_mode in range(2):
                         err_Martin_ = calculate_radius_directed_error(sig_Martin[i_mode,0:t_stop],R2)
                         err_Martin.append(mlab.movavg(abs(err_Martin_),movavg_taps))
                    try : 
                        ber_martin = calculate_BER(sig_Martin,range(t_conv,t_stop))
                    except:
                        ber_martin = [1,1]
                    if produce_plots:
                        title = "Martin_mu" + str(mu_m)+"_lb"+str(lb)
                        plot_constellation(sig_Martin[:,t_conv:t_stop],title,True,mimodir)        
                        plot_error(err_Martin,title,True,mimodir)
                        plot_taps(taps_Martin[:,:,0,:],True,mimodir)
                    final_error_Martin = calculate_final_error(err_Martin,t_conv,t_stop)
                    t_conv_martin = calculate_convergence(err_Martin,final_error_Martin)
                    df_martin = UpdateDataFrame(pmd,snr,mu_m,lb,ber_martin,t_conv_martin,final_error_Martin,r,"Martin")
                    df = df.append(df_martin)

df = df.sort_values(['MODE','PMD','SNR','BER'],ascending=[0,0,1,1])
print(df)
df.to_csv(testDir + "\Results.csv", sep=',')


