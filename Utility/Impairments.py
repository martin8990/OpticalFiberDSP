
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from sklearn.preprocessing import normalize
def build_mixing_matrix(nmodes,mixing,loss):
    matrix = np.ones(nmodes * nmodes) * mixing
    matrix = matrix.reshape(nmodes,nmodes)
    for i_mode in range(nmodes):
          matrix[i_mode,i_mode] = 1-loss
    # Normalize

    normed_matrix = normalize(matrix, axis=1, norm='l1')
    return normed_matrix

def apply_impulse_response_impairment(sig):
    ntaps = 200
    taps = np.load("impulse_response.npy")[8191-ntaps:8191+ntaps+1]
    taps =taps #+ 1j*np.zeros_like(taps)
    nmodes = len(sig[:,0])
    sig_out = sig.copy()
    for i in range(nmodes):
        sig_out[i,:] = convolve(taps,sig[i,:])[0:len(sig[i,:])]
    return sig_out

def CreateFrequencySpectrum(NSamps,f_s):
    N = NSamps
    f_pos = np.arange(0,N/2,1)
    f_neg = np.arange(N/2,0 ,-1)
    f = np.concatenate([f_pos,f_neg])
    return f * f_s/N

def applyDelay(sig:np.array,delay,f):
     H = np.exp(-1j * delay * f)
     X = np.fft.fft(sig)
     Y = X/H
     return np.fft.ifft(Y)
     
def apply_mltichnl_delayed_matrix_impairment(sig : np.ndarray,delay : int,matrix):

    impaired_sig = np.zeros_like(sig)
    for i_mode in range(sig.shape[0]):
        impaired_sig[i_mode] = sig[i_mode]
    impaired_sig[1] = np.roll(sig[1],delay)
        
    for k in range(sig.shape[1]):
        impaired_sig[:,k] = np.matmul(impaired_sig[:,k],matrix)
    return impaired_sig

def apply_2chnl_delayed_matrix_impairment(sig : np.ndarray,delay : int,matrix):
    if sig.shape[0] != 2:
        raise ValueError('Only works for two channels/modes!')
    impaired_sig = np.zeros_like(sig)
    impaired_sig[1] = np.roll(sig[1],delay)
    impaired_sig[0] = sig[0]
        
    for k in range(sig.shape[1]):
        impaired_sig[:,k] = np.matmul(impaired_sig[:,k],matrix)
    return impaired_sig
        
def apply_2chnl_delayed_after_mix_matrix_impairment(sig : np.ndarray,t_delay,matrix):
    if sig.shape[0] != 2:
        raise ValueError('Only works for two channels/modes!')
   
    impaired_sig = np.zeros_like(sig)
    
    for k in range(sig.shape[1]):
        impaired_sig[:,k] = np.matmul(sig[:,k],matrix)
    impaired_sig[1] = np.roll(impaired_sig[1],t_delay)
    return impaired_sig    


if __name__ == "__main__":
    ntaps = 200
    taps = np.load("impulse_response.npy")[8191-ntaps:8191+ntaps+1]
    plt.figure()
    plt.plot(taps)
    plt.xlabel("Tap (-)")
    plt.ylabel("Weight (-)")
    plt.grid(1)
    apply_2chnl_delayed_matrix_impairment(np.ones(100).reshape(2,50),1,1)
    
    #plt.show()


    
