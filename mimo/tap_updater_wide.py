import numpy as np
from mimo.mimo import WideBlockDistributer

class WideTapUpdater():
    def save_timedomain_taps(self, H, lb, nmodes, ovsmpl):
        for i_input in range(nmodes):
            for i_output in range(nmodes):
                for i_ovsmpl in range(ovsmpl):
                    for i_wide in [0,1]:
                        myrange = range(i_ovsmpl,lb*ovsmpl + i_ovsmpl,ovsmpl)
                        myTaps =np.fft.ifft(H[i_input,i_output,i_ovsmpl,i_wide])[:lb]
                        self.h_saved[i_input,i_output,i_wide,self.i_block,myrange] = myTaps[::-1]
        self.i_block += 1
    
    def __init__(self,mu,block_distr : WideBlockDistributer):
        self.mu = mu
        
        nmodes = block_distr.nmodes
        lb = block_distr.lb
        ovsmpl = block_distr.ovsmpl
 
        h = np.zeros((nmodes,nmodes,ovsmpl,2,lb * 2),dtype = np.complex128)
        H = np.zeros_like(h)

        CTap = np.int(lb / 2)
        for i_output in range(nmodes):
            h[i_output,i_output,0,0,CTap] = 0.5 + 0j
            h[i_output,i_output,0,1,CTap] = 0.5 + 0j
        for i_input in range(nmodes):
            for i_output in range(nmodes):
                for i_ovsmpl in range(ovsmpl):
                    for i_wide in [0,1]:
                        H[i_input,i_output,i_ovsmpl,i_wide] = np.fft.fft(h[i_input,i_output,i_ovsmpl,i_wide])
        self.H = H
        self.h = h[:,:,:,:,:lb]
        self.h_saved = np.zeros((nmodes,nmodes,2,block_distr.nblocks,lb*ovsmpl),dtype = np.complex128)
        self.i_block = 0
        self.save_timedomain_taps(H,lb,nmodes,ovsmpl)

    def retrieve_timedomain_taps(self):
        return self.h_saved

    def update_taps(self,block_distr : WideBlockDistributer):
         raise ValueError("Pick a Tapupdater")

class WideFrequencyDomainTapUpdater(WideTapUpdater):
    def update_taps(self,block_distr : WideBlockDistributer):

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
                    for i_wide in [0,1]:
                        s_ = np.fft.ifft(np.conj(block_fd[i_input,i_ovsmpl,i_wide]) * E)[:lb]
                        H[i_input,i_output,i_ovsmpl,i_wide] += mu * np.fft.fft(np.append(s_,zeros))
        self.H = H
        self.save_timedomain_taps(H, lb, nmodes, ovsmpl)


