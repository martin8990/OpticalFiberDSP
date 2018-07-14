import numpy as np
from mimo.mimo import BlockDistributer

class TapUpdater():
    def save_timedomain_taps(self, H, lb, nmodes, ovsmpl,ovconj):
        """ For vizualisation purposes""" 
        for i_input in range(nmodes):
            for i_output in range(nmodes):
                for i_ovsmpl in range(ovsmpl):
                    for i_ovconj in range(ovconj):
                        myrange = range(i_ovsmpl,lb*ovsmpl + i_ovsmpl,ovsmpl)
                        myTaps =np.fft.ifft(H[i_input,i_output,i_ovsmpl,i_ovconj])[:lb]
                        self.h_saved[i_input,i_output,i_ovconj,self.i_block,myrange] = myTaps[::-1]
        self.i_block += 1
    
    def __init__(self,mu,block_distr : BlockDistributer):
        """ 
        Parameters
        ----------
        mu : stepsize, ussually between 3e-3 and 5e-5, 
        I would recommend using a higher step size for noisy signals while using
        a higher one for less noise ones.
        """
        self.mu = mu
        nmodes = block_distr.nmodes
        lb = block_distr.lb
        ovsmpl = block_distr.ovsmpl
        ovconj = block_distr.ovconj
        h = np.zeros((nmodes,nmodes,ovsmpl,2,lb * 2),dtype = np.complex128)
        H = np.zeros_like(h)

        CTap = np.int(lb / 2)
        for i_output in range(nmodes):
            for i_ovconj in range(ovconj):
                h[i_output,i_output,0,i_ovconj,CTap] = 1/ovconj + 0j
                # I decided to leave the odd sampled taps at 0
        for i_input in range(nmodes):
            for i_output in range(nmodes):
                for i_ovsmpl in range(ovsmpl):
                    for i_ovconj in range(ovconj):
                        H[i_input,i_output,i_ovsmpl,i_ovconj] = np.fft.fft(h[i_input,i_output,i_ovsmpl,i_ovconj])
        self.H = H
        self.h = h[:,:,:,:,:lb]
        self.h_saved = np.zeros((nmodes,nmodes,ovconj,block_distr.nblocks,lb*ovsmpl),dtype = np.complex128)
        self.i_block = 0
        self.save_timedomain_taps(H,lb,nmodes,ovsmpl,ovconj)

    def retrieve_timedomain_taps(self):
        return self.h_saved

    def update_taps(self,block_distr : BlockDistributer):
         raise ValueError("Pick a Tapupdater")

class FrequencyDomainTapUpdater(TapUpdater):
    def update_taps(self,block_distr : BlockDistributer):
        """Blockwize Frequency domain tap updater after [1]"""
        H = self.H
        nmodes = block_distr.nmodes
        ovsmpl = block_distr.ovsmpl
        ovconj = block_distr.ovconj
        lb = block_distr.lb
        mu = self.mu
        e = block_distr.block_error
        zeros = np.zeros(lb,dtype = np.complex128)
        block_fd = block_distr.double_block_fd

        for i_output in range(nmodes):
            E = np.fft.fft(np.append(zeros,e[i_output,:]))
            for i_input in range(nmodes):        
                for i_ovsmpl in range(ovsmpl):
                    for i_ovconj in range(ovconj):
                        s_ = np.fft.ifft(np.conj(block_fd[i_input,i_ovsmpl,i_ovconj]) * E)[:lb]
                        H[i_input,i_output,i_ovsmpl,i_ovconj] += mu * np.fft.fft(np.append(s_,zeros))
        self.H = H
        self.save_timedomain_taps(H, lb, nmodes, ovsmpl,ovconj)


class TimedomainTapupdater(TapUpdater):
    def update_taps(self,block_distr : BlockDistributer):
        """
        Time domain tap updater after the Hybrid implementation in [1], unfortunately not functioning as desired
        """
        h = self.h
        nmodes = block_distr.nmodes
        ovsmpl = block_distr.ovsmpl
        ovconj = block_distr.ovconj
        lb = block_distr.lb
        
        e = block_distr.block_error
        zeros = np.zeros(lb,dtype = np.complex128)
        block = block_distr.double_block[:,:,:,lb:]
        #print(block.shape)
        for i_output in range(nmodes):
            for i_input in range(nmodes):        
                for i_ovsmpl in range(ovsmpl):
                    for i_ovconj in range(ovconj):
                        shape_H = (i_input,i_output,i_ovsmpl,i_ovconj) 
                        h[shape_H] += mu * np.conj(block[i_input,i_ovsmpl,i_ovconj]) * e[i_output]
                        self.H[shape_H] = np.fft.fft(np.append(h[shape_H],zeros))
        self.h = h
        self.save_timedomain_taps(self.H, lb, nmodes, ovsmpl,ovconj)


