import numpy as np
from mimo.mimo import BlockDistributer

class TapUpdater():
    def save_timedomain_taps(self, H, lb, nmodes, ovsmpl):
        for i_input in range(nmodes):
            for i_output in range(nmodes):
                for i_ovsmpl in range(ovsmpl):
                    myrange = range(i_ovsmpl,lb*ovsmpl + i_ovsmpl,ovsmpl)
                    myTaps =np.fft.ifft(H[i_input,i_output,i_ovsmpl])[:lb]
                    self.h_saved[i_input,i_output,self.i_block,myrange] = myTaps[::-1]
        self.i_block += 1
    
    def __init__(self,mu,block_distr : BlockDistributer):
        self.mu = mu
        
        nmodes = block_distr.nmodes
        lb = block_distr.lb
        ovsmpl = block_distr.ovsmpl
 
        h = np.zeros(lb * 2 * nmodes * nmodes * ovsmpl,dtype = np.complex128).reshape(nmodes,nmodes,ovsmpl,lb * 2)
        H = np.zeros_like(h)

        CTap = np.int(lb / 2)
        for i_output in range(nmodes):
            h[i_output,i_output,0,CTap] = 1 + 0j
        for i_input in range(nmodes):
            for i_output in range(nmodes):
                for i_ovsmpl in range(ovsmpl):
                    H[i_input,i_output,i_ovsmpl] = np.fft.fft(h[i_input,i_output,i_ovsmpl])
        self.H = H
        self.h = h[:,:,:,:lb]
        self.h_saved = np.zeros(lb * nmodes * nmodes * ovsmpl * block_distr.nblocks,dtype = np.complex128)
        self.h_saved = self.h_saved.reshape(nmodes,nmodes,block_distr.nblocks,lb*ovsmpl)
        self.i_block = 0
        self.save_timedomain_taps(H,lb,nmodes,ovsmpl)

    def retrieve_timedomain_taps(self):
        return self.h_saved

    def update_taps(self,block_distr : BlockDistributer):
         raise ValueError("Pick a Tapupdater")

class FrequencyDomainTapUpdater(TapUpdater):
    def update_taps(self,block_distr : BlockDistributer):
        H = self.H
        nmodes = block_distr.nmodes
        ovsmpl = block_distr.ovsmpl
        lb = block_distr.lb
        mu = self.mu
        e = block_distr.block_error
        zeros = np.zeros(lb,dtype = np.complex128)
        block_fd = block_distr.double_block_fd
        for i_output in range(nmodes):
            E = np.fft.fft(np.append(zeros,e[i_output,:]))
            for i_input in range(nmodes):        
                for i_ovsmpl in range(ovsmpl):
                    s_ = np.fft.ifft(np.conj(block_fd[i_input,i_ovsmpl]) * E)[:lb]
                    H[i_input,i_output,i_ovsmpl] = H[i_input,i_output,i_ovsmpl] + mu * np.fft.fft(np.append(s_,zeros))
        self.H = H
        self.save_timedomain_taps(H, lb, nmodes, ovsmpl)


class TimeDomainTapUpdater(TapUpdater):
    def update_taps(self,block_distr : BlockDistributer):
        h = self.h
        nmodes = block_distr.nmodes
        ovsmpl = block_distr.ovsmpl
        lb = block_distr.lb
        mu = self.mu
        block_td = block_distr.block
        e =block_distr.block_error
        zeros = np.zeros(lb,dtype = np.complex128)
        H = self.H

        for i_output in range(nmodes):
            for i_input in range(nmodes):        
                for i_ovsmpl in range(ovsmpl):
                    h[i_input,i_output,i_ovsmpl] += (mu * np.conj(block_td[i_input,i_ovsmpl]) * e[i_output])
                    H[i_input,i_output,i_ovsmpl] = np.fft.fft(np.append(h[i_input,i_output,i_ovsmpl],zeros)) 
        self.h = h
        self.H = H
        self.save_timedomain_taps(H, lb, nmodes, ovsmpl)
 

        

