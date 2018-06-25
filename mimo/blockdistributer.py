
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

def test_inputs(sig : np.ndarray,lb,mu):
    if sig.dtype != np.complex128:
        raise ValueError("Only complex128 input Allowed")
    if lb < 64 :
        print("Blocklength " + str(lb) + " might be too low, this might hinder mode dispersion compensation")
    if lb > 256 :
        print ("Blocklength " + str(lb) + " might be too high, this decreases tolerance to polarization rotaions")
    if mu < 1e-4 :
        print ("mu " + str(mu) + " might be too low to track the channel")
    if mu > 1e-2 :
        print ("mu " + str(mu) + " might be too high which may lead to instabillity")


class BlockDistributer():


    def _get_block_range(self,i_block, lb):
        return range(i_block * lb - lb,i_block * lb)

    def _get_double_block_range(self,i_block, lb):
        return range(i_block * lb - lb,i_block * lb + lb)

    def _separate_oversampled_samples(self,sig):
        nmodes = sig.shape[0]
        nsamps = sig.shape[1]
        ovsmpl = self.ovsmpl
        sig_separated = np.zeros_like(sig).reshape(nmodes,ovsmpl,int(nsamps/ovsmpl))
        for i_ovsmpl in range(ovsmpl):
            for i_input in range(nmodes):
                sig_separated[i_input,i_ovsmpl] = sig[i_input,range(i_ovsmpl,nsamps,ovsmpl)]
        return sig_separated

    def reselect_blocks(self,i_block):
        nmodes= self.nmodes
        ovsmpl = self.ovsmpl
        lb = self.lb
        range_block = self._get_block_range(i_block,lb)
        range_double_block = self._get_double_block_range(i_block,lb)
        self.double_block = self.sig_separated[:,:,range_double_block]
        self.block = self.sig_separated[:,:,range_block]
        double_block_fd = np.zeros_like(self.double_block)
        for i_input in range(double_block_fd.shape[0]):
            for i_ovsmpl in range(double_block_fd.shape[1]):
                double_block_fd[i_input,i_ovsmpl] = np.fft.fft(self.double_block[i_input,i_ovsmpl])
        self.double_block_fd = double_block_fd
        self.range_block = range_block
        self.i_block = i_block
           
    def insert_compensated_block(self,block_compensated):
        self.block_compensated = block_compensated
        self.sig_compensated[:,self.range_block] = block_compensated
    

    def insert_block_error(self,block_error):
        self.block_error = block_error

    def recalculate_shifted_fd_block(self,shift):
        i_block = self.i_block
        lb = self.lb

        if i_block > 1:
            inputrange = range(i_block * lb - lb + shift,i_block * lb + lb + shift)
            double_block_fd = np.zeros_like(self.double_block)
            for i_input in range(double_block_fd.shape[0]):
                for i_ovsmpl in range(double_block_fd.shape[1]):
                    double_block_fd[i_input,i_ovsmpl] = np.fft.fft(self.sig_separated[i_input,i_ovsmpl,inputrange])
            self.double_block_fd = double_block_fd
           
    def __init__(self,sig,lb,ovsmpl):
        self.lb = lb
        self.ovsmpl = ovsmpl
        nmodes = sig.shape[0]
        nsyms = int(sig.shape[1]/ovsmpl)
        self.nmodes = nmodes
        self.sig_separated = self._separate_oversampled_samples(sig)
        self.sig_compensated = np.zeros(nmodes*nsyms,dtype=np.complex128).reshape(nmodes,nsyms)
        self.nblocks = int(nsyms/lb)
        self.phases = np.zeros((nmodes,lb))

class WideBlockDistributer(BlockDistributer):
       
    def _separate_oversampled_samples(self,sig):
        nmodes = sig.shape[0]
        nsamps = sig.shape[1]
        ovsmpl = self.ovsmpl
        dims = 2
        shp = (nmodes,ovsmpl,dims,int(nsamps/ovsmpl))
        sig_separated = np.zeros(shp,dtype=np.complex128)
        for i_ovsmpl in range(ovsmpl):
            for i_input in range(nmodes):
                sig_separated[i_input,i_ovsmpl,0] = sig[i_input,range(i_ovsmpl,nsamps,ovsmpl)]
                sig_separated[i_input,i_ovsmpl,1] = np.conj(sig_separated[i_input,i_ovsmpl,0])
        return sig_separated

    def reselect_blocks(self,i_block):
        nmodes= self.nmodes
        ovsmpl = self.ovsmpl
        lb = self.lb
        range_block = self._get_block_range(i_block,lb)
        range_double_block = self._get_double_block_range(i_block,lb)
        self.double_block = self.sig_separated[:,:,:,range_double_block]
        self.block = self.sig_separated[:,:,:,range_block]
        double_block_fd = np.zeros_like(self.double_block)
        for i_input in range(double_block_fd.shape[0]):
            for i_ovsmpl in range(double_block_fd.shape[1]):
                for i_wide in [0,1]:
                    double_block_fd[i_input,i_ovsmpl,i_wide] = np.fft.fft(self.double_block[i_input,i_ovsmpl,i_wide])
        self.double_block_fd = double_block_fd
        self.range_block = range_block
        self.i_block = i_block           
    
#Todo overwrite shifted FD
           
    def __init__(self,sig,lb,ovsmpl):
        super().__init__(sig,lb,ovsmpl)
