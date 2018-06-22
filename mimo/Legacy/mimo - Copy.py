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
from mimo.error_calculator import *
from mimo.tap_updater import *
from mimo.tap_updater_td import *
from mimo.compensator import *

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

class BlockwizeMimo():
    def __init__(self,nmodes,is_hybrid = False,lb = 80,ovsmpl = 1,mu = 3e-4,error_calculator = CMAErrorCalculator()):
        self.compensator = Compensator(nmodes,lb,ovsmpl)
        self.nmodes = nmodes
        self.lb = lb
        self.ovsmpl = ovsmpl
        self.mu = mu
        self.error_calculator = error_calculator
        self.is_hybrid = is_hybrid
        print(is_hybrid)
        if is_hybrid:
            self.tap_updater = TimeDomainTapUpdater(lb,mu,nmodes,ovsmpl)
        else :
            self.tap_updater = TapUpdater(lb,mu,nmodes,ovsmpl)


    def _get_block_range(self,i_block, lb):
        return range(i_block * lb - lb,i_block * lb)

    def _get_double_block_range(self,i_block, lb):
        return range(i_block * lb - lb,i_block * lb + lb)

    def _separate_oversampled_samples(self,sig):
        nmodes = self.nmodes
        nsamps = sig.shape[1]
        ovsmpl = self.ovsmpl
        sig_separated = np.zeros_like(sig).reshape(nmodes,ovsmpl,int(nsamps/ovsmpl))
        for i_ovsmpl in range(ovsmpl):
            for i_input in range(nmodes):
                sig_separated[i_input,i_ovsmpl] = sig[i_input,range(i_ovsmpl,nsamps,ovsmpl)]
        return sig_separated
    
    def equalize_signal(self,sig,store_taps = False):

        nmodes = self.nmodes
        ovsmpl = self.ovsmpl
        lb = self.lb
        nsyms = int(sig.shape[1]/ovsmpl)
        print(nsyms)
            
        compensator = self.compensator
        error_calculator = self.error_calculator
        tap_updater = self.tap_updater

        sig_separated = self._separate_oversampled_samples(sig)
        sig_compensated = np.zeros(nmodes*nsyms,dtype=np.complex128).reshape(nmodes,nsyms)

        nblocks = int( nsyms/lb)
        H = tap_updater.H

        test_inputs(sig,lb,self.mu)
        tap_saver = None
        if store_taps:
            tap_saver = TapsSaver(nblocks,nmodes,ovsmpl,lb) 

        for i_block in range(1,nblocks):
            # Refactor selection process later
            range_block = self._get_block_range(i_block,lb)
            range_double_block = self._get_double_block_range(i_block,lb)

            double_block_separated = sig_separated[:,:,range_double_block]
            block_compensated,block_fd = compensator.compensate(double_block_separated,H)

            sig_compensated[:,range_block] = block_compensated
            block_error = error_calculator.calculate_error(block_compensated)
            
            if self.is_hybrid:
                H = tap_updater.update_taps(double_block_separated,block_error)
            else:                
                H = tap_updater.update_taps(block_fd,block_error)
            if store_taps:
                tap_saver.save_taps(i_block,tap_updater.retrieve_timedomain_taps())

        if store_taps:
            return sig_compensated,tap_saver.h_stored
        else:    
            return sig_compensated, tap_updater.retrieve_timedomain_taps()   

    # At this level we mainly select the blocks from the signal and manage the parts that make up the mimo
     


if __name__ == '__main__':
    nmodes = 1
    ovsmpl = 1
    nsyms = 1000
    lb = 10
    error_calculator = CMAErrorCalculator()
    mimo = FrequencyDomainBlockwizeMimo(nmodes,lb,ovsmpl,1e-4, error_calculator)
    sig = np.ones(nmodes*ovsmpl*nsyms).reshape(nmodes,nsyms*ovsmpl)
    sig,H =  mimo.equalize_signal(sig)
    
    
    print(sig)
