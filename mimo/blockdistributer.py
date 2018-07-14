# Author M.C. van Leeuwen (m.c.v.leeuwen@student.tue.nl)

# sig : signal, a numpy ndarray where dim0 represents the channels and dim1 the samples
# mu : stepsize
# lb : blocklength

import numpy as np
from mimo.trainer import Trainer


# The blockdistibuter exists to provide signal blocks to the components 
# of the mimo. This makes the components cleaner as they dont have to define the ranges on their own .
# It also contains the block settings, it therefore avoids
# dataclumps in the code. 

class BlockDistributer():

    def __init__(self,sig,lb,ovsmpl,ovconj,trainer : Trainer):

        self.lb = lb
        self.ovsmpl = ovsmpl
        nmodes = sig.shape[0]
        nsyms = int(sig.shape[1]/ovsmpl)
        self.nmodes = nmodes
        self.ovconj = ovconj
        self.sig_separated = self._separate_samples(sig)
        self.sig_compensated = np.zeros(nmodes*nsyms,dtype=np.complex128).reshape(nmodes,nsyms)
        self.nblocks = int(nsyms/lb)
        self.phases = np.zeros((nmodes,lb))
        self.trainer = trainer
        
    def _get_block_range(self,i_block, lb):
        return np.arange(i_block * lb - lb,i_block * lb)

    def _get_double_block_range(self,i_block, lb):
        return np.arange(i_block * lb - lb,i_block * lb + lb)

    def insert_compensated_block(self,block_compensated):
        self.block_compensated = block_compensated
        self.sig_compensated[:,self.range_block] = block_compensated

    def insert_block_error(self,block_error):
        self.block_error = block_error
    
    # :    
    def _separate_samples(self,sig):
        """to more easily use the oversampled and widely linear mimo"""
        nmodes = sig.shape[0]
        nsamps = sig.shape[1]
        ovsmpl = self.ovsmpl
        ovconj = self.ovconj
        shp = (nmodes,ovsmpl,ovconj,int(nsamps/ovsmpl))
        sig_separated = np.zeros(shp,dtype=np.complex128)
        for i_ovsmpl in range(ovsmpl):
            for i_input in range(nmodes):
                sig_separated[i_input,i_ovsmpl,0] = sig[i_input,range(i_ovsmpl,nsamps,ovsmpl)]
                if self.ovconj>1:
                    sig_separated[i_input,i_ovsmpl,1] = np.conj(sig_separated[i_input,i_ovsmpl,0])
        return sig_separated

    
    def reselect_blocks(self,i_block):
        nmodes= self.nmodes
        lb = self.lb

        range_block = self._get_block_range(i_block,lb)
        range_double_block = self._get_double_block_range(i_block,lb)
    
        self.double_block = self.sig_separated[:,:,:,range_double_block]
        self.block = self.sig_separated[:,:,:,range_block]
        self.trainer.update_block(i_block,range_block)
        
        double_block_fd = np.zeros_like(self.double_block)
        for i_input in range(double_block_fd.shape[0]):
            for i_ovsmpl in range(double_block_fd.shape[1]):
                for i_ovconj in range(ovconj):
                    double_block_fd[i_input,i_ovsmpl,i_ovconj] = np.fft.fft(self.double_block[i_input,i_ovsmpl,i_ovconj])
        
        self.double_block_fd = double_block_fd
        self.range_block = range_block
        self.i_block = i_block           
    
    # 
    def shift_fd_block(self,shift):
        """Compensates the shift/delay introduced by phaserecovery, 
             so that the taps are updated with the correct input samples"""
        i_block = self.i_block
        lb = self.lb
        ovconj = self.ovconj

        if i_block > 1: # First block is out of range
            inputrange = self._get_double_block_range(i_block,lb) + shift 
            double_block_fd = np.zeros_like(self.double_block)
            for i_input in range(double_block_fd.shape[0]):
                for i_ovsmpl in range(double_block_fd.shape[1]):
                    for i_ovconj in range(ovconj):
                        block_dim = np.fft.fft(self.sig_separated[i_input,i_ovsmpl,i_ovconj,inputrange])
                        double_block_fd[i_input,i_ovsmpl,i_ovconj] = block_dim
            self.double_block_fd = double_block_fd
           
