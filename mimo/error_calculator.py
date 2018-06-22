# Author M.C. van Leeuwen (m.c.v.leeuwen@student.tue.nl)
# Last Updated : 15-6-18
# sig : signal, a numpy ndarray where dim0 represents the channels and dim1 the samples
# fd : frequency domain
# bw : block-wise
# cma : constant modulus algorithm
# lms : least-mean-square algorithm
# mu : stepsize
# lb : blocklength
import numpy as np
from mimo.mimo import BlockDistributer
import matplotlib.mlab as mlab

class MimoErrorCalculator(): 
    def __init__(self,block_distr : BlockDistributer):
        nmodes = block_distr.nmodes
        lb = block_distr.lb
        nblocks = block_distr.nblocks
        size = nmodes * lb * block_distr.nblocks
        self.error = np.zeros(size).reshape(nmodes,nblocks*lb)
        self.i_block = 0
        self.nmodes = block_distr.nmodes

    def start_error_calculation(self,block_distr : BlockDistributer):
        block = block_distr.block_compensated
        err = self.calculate_error(block)
        block_distr.insert_block_error(err)
        self.error[:,self.i_block*block_distr.lb:(self.i_block+1)*block_distr.lb] = err
        self.i_block+=1
    
    def calculate_error(self,block) -> np.ndarray:
        raise ValueError('please select one of the errorCalculators that derives from this class')
    
    def retrieve_error(self):
        err_Martin = []
        movavg_taps = 1000
        for i_mode in range(self.nmodes):
            err_Martin.append(mlab.movavg(abs(self.error[i_mode]),movavg_taps))
        return err_Martin

        
class CMAErrorCalculator(MimoErrorCalculator):
    def calculate_error(self,block):
        return (1 - block * np.conj(block)) * block

        
class TrainedLMS(MimoErrorCalculator):
    def __init__(self,trainingSyms : np.ndarray, constellation : list, block_distr : BlockDistributer,n_trainingsyms,offset):
        self.trainingSyms = trainingSyms
        self.constellation = constellation
        self.n_training_syms = n_trainingsyms
        self.counter = block_distr.lb
        self.lb = block_distr.lb
        self.offset = offset
        super().__init__(block_distr)

    def AddTrainingLoops(self,sig : np.ndarray,ovsmpl,nloops : int):
        n_training_syms = self.n_training_syms
        training_samps = sig[:,:ovsmpl * n_training_syms]
        sig_with_loops = sig.copy()
        
        for i_loop in range(nloops):
            sig_with_loops = np.append(training_samps,sig_with_loops,axis = 1)
            self.trainingSyms = np.append(self.trainingSyms,self.trainingSyms,axis = 1)
        self.n_training_syms = n_training_syms + (nloops * n_training_syms)
        return sig_with_loops

    # Insert one block for each input
    def calculate_error(self,block):
        lb = self.lb
        #offset = -int(lb/2)
        if self.counter < self.n_training_syms:
            trainingRange = range(self.counter + self.offset,self.counter + lb + self.offset)
            e = self.trainingSyms[:,trainingRange]-block
        else :
            e = np.zeros_like(block)
            # TODO : Convert to matrix implementation
            for i_output in range(block.shape[0]):
                for k in range(lb):
                    best_sym = np.argmin(np.abs( self.constellation-block[i_output,k])) 
                    e[i_output,k] =  self.constellation[best_sym] - block[i_output,k]
        self.counter = self.counter + lb  
        return e
