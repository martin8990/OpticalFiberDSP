# Author M.C. van Leeuwen (m.c.v.leeuwen@student.tue.nl)
# Last Updated : 25-5-18
# sig : signal, a numpy ndarray where dim0 represents the channels and dim1 the samples
# fd : frequency domain
# bw : block-wise
# cma : constant modulus algorithm
# lms : least-mean-square algorithm
# mu : stepsize
# lb : blocklength


import numpy as np
from timeit import default_timer as timer

class MimoErrorCalculator(): 
    def calculate_error(self,sig,i_block):
        raise ValueError('please select one of the errorCalculators that derives from this class')
        return sig
class CMAErrorCalculator(MimoErrorCalculator):
    def calculate_error(self,sig,i_block):
        return (1 - sig * np.conj(sig)) * sig

class TrainedLMS(MimoErrorCalculator):
    def __init__(self, trainingSyms,offset,constellation : list,n_training_syms : int):
        self.trainingSyms = trainingSyms
        self.offset = offset
        self.constellation = constellation
        self.n_training_syms = n_training_syms

    def calculate_error(self,sig,i_block):
        lb = sig.shape[0]
        if i_block*lb < self.n_training_syms:
            trainingRange = range(i_block * lb + self.offset,i_block * lb + lb + self.offset)
            e = self.trainingSyms[trainingRange]-sig
        else :
            e = np.zeros_like(sig)
            for k in range(lb):
                best_sym = np.argmin(np.abs( self.constellation-sig[k])) 
                e[k] =  self.constellation[best_sym] - sig[k]
        return e


def _setup_initial_taps(nmodes, lb, ovsmpl):
    h = np.zeros(lb * 2 * nmodes * nmodes,dtype = np.complex128).reshape(nmodes,nmodes,lb * 2)
    H = np.zeros(lb * 2 * nmodes * nmodes * ovsmpl,dtype = np.complex128).reshape(nmodes,nmodes,ovsmpl,lb * 2)
    
    CTap = np.int(lb / 2)
    for i_output in range(nmodes):
        h[i_output,i_output,CTap] = 1/ovsmpl + 0j
    for i_input in range(nmodes):
        for i_output in range(nmodes):
            for i_ovsmpl in range(ovsmpl):
                H[i_input,i_output,i_ovsmpl,:] = np.fft.fft(h[i_input,i_output,:])
    return H

def _retrieve_timedomain_taps(H):
    nmodes = H.shape[0]
    ovsmpl = H.shape[2]
    lb = int(H.shape[3]/2)
    h = np.zeros_like(H)
    for i_input in range(nmodes):
        for i_output in range(nmodes):
            for i_ovsmpl in range(ovsmpl):
                h[i_input,i_output,i_ovsmpl] = np.fft.ifft(H[i_input,i_output,i_ovsmpl])
    return h[:,:,:,0:lb]

def _get_block_range(i_block, lb):
    return range(i_block * lb - lb,i_block * lb)

def _get_double_block_range(i_block, lb):
    return range(i_block * lb - lb,i_block * lb + lb)


def _compensate(sig,H):
    nmodes = sig.shape[0]
    ovsmpl = sig.shape[1]
    lb = int(sig.shape[2]/2)
    second_block = range(lb,2 * lb)
    
    sig_fd = np.zeros(nmodes* lb * 2 * ovsmpl,dtype = np.complex128).reshape(nmodes,ovsmpl,lb*2)
    sig_compensated = np.zeros(nmodes * lb ,dtype = np.complex128).reshape(nmodes,lb)

    for i_input in range(nmodes):
        for i_output in range(nmodes):        
            FD_temp = np.zeros(lb*2,dtype = np.complex128)  
            for i_ovsmpl in range(ovsmpl):
                sig_fd[i_input,i_ovsmpl] = np.fft.fft(sig[i_input,i_ovsmpl])  
                FD_temp = FD_temp + sig_fd[i_input,i_ovsmpl] * H[i_input,i_output,i_ovsmpl]
            sig_compensated[i_output] = sig_compensated[i_output] + np.fft.ifft(FD_temp)[second_block]
    return sig_compensated,sig_fd






def _update_taps(sig_compensated,sig_fd,H,i_block,errorCalculator : MimoErrorCalculator,mu):
    nmodes = H.shape[0]
    ovsmpl = H.shape[2]
    lb = int(H.shape[3]/2)
    zeros = np.zeros(lb,dtype = np.complex128)
    first_block = range(lb)
    
    for i_input in range(nmodes):
        for i_output in range(nmodes):        
            for i_ovsmpl in range(ovsmpl):                
                e = errorCalculator.calculate_error(sig_compensated[i_output],i_block)
                E = np.fft.fft(np.append(zeros,e)) 
                s_ = np.fft.ifft(np.conj(sig_fd[i_input,i_ovsmpl]) * E)[first_block]
                H[i_input,i_output,i_ovsmpl] = H[i_input,i_output,i_ovsmpl] + mu * np.fft.fft(np.append(s_,zeros))
    return H


def _separate_oversampled_samples(sig,ovsmpl):
    nmodes = sig.shape[0]
    nsamps = sig.shape[1]
    sig_separated = np.zeros_like(sig).reshape(nmodes,ovsmpl,int(nsamps/ovsmpl))
    for i_ovsmpl in range(ovsmpl):
        for i_input in range(nmodes):
            sig_separated[i_input,i_ovsmpl] = sig[i_input,range(i_ovsmpl,nsamps,ovsmpl)]
    return sig_separated

def mimo_fd_ba(sig,error_calculator = CMAErrorCalculator(),lb = 80,mu = 3e-4,ovsmpl = 1):
    
    nmodes = sig.shape[0]
    nsyms = int(sig.shape[1]/ovsmpl)
    sig_separated = _separate_oversampled_samples(sig,ovsmpl)
    sig_compensated = np.zeros(nmodes*nsyms,dtype=np.complex128).reshape(nmodes,nsyms)
    print(sig_compensated.shape[0])
    H = _setup_initial_taps(nmodes,lb,ovsmpl)
    nblocks = int( nsyms/lb)

    for i_block in range(1,nblocks):
        range_block = _get_block_range(i_block,lb)
        range_double_block = _get_double_block_range(i_block,lb)
        sig_compensated[:,range_block],sig_fd = _compensate(sig_separated[:,:,range_double_block],H)
        H = _update_taps(sig_compensated[:,range_block],sig_fd,H,i_block,error_calculator,mu)
    #if set.ovsmpl > 0:
    #    d.sig = d.sig[:,:d.NSyms]    


    return sig_compensated, _retrieve_timedomain_taps(H)    


if __name__ == '__main__':
    nmodes = 1
    ovsmpl = 1
    nsyms = 1000
    sig = np.ones(nmodes*ovsmpl*nsyms).reshape(nmodes,nsyms*ovsmpl)
    
    out,H = mimo_fd_ba(sig)

    print(out)
