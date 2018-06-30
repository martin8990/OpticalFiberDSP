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
from mimo.mimo import BlockDistributer, Trainer
import matplotlib.mlab as mlab

class MimoErrorCalculator(): 
    def __init__(self,block_distr : BlockDistributer):
        nmodes = block_distr.nmodes
        lb = block_distr.lb
        nblocks = block_distr.nblocks
        self.error = np.zeros((nmodes , lb *nblocks),dtype=np.complex128)
        self.i_block = 0
        self.nmodes = block_distr.nmodes

    def start_error_calculation(self,block_distr : BlockDistributer,trainer : Trainer):
        block = block_distr.block_compensated
        err = self.calculate_error(block,trainer)
        err = err*np.exp(1.j*-block_distr.phases)
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
    def calculate_error(self,block,trainer):
        return (1 - block * np.conj(block)) * block

        
class TrainedLMS(MimoErrorCalculator):
    def __init__(self,block_distr : BlockDistributer):
        super().__init__(block_distr)

    # Insert one block for each input
    def calculate_error(self,block,trainer : Trainer):
        if trainer.in_training:
            e = trainer.block_sequence-block
        else :
            e = np.zeros_like(block)
            for i_output in range(block.shape[0]):
                best_sym = np.argmin(np.abs( trainer.constellation-block[i_output,:,np.newaxis]),axis=1) 
                for k in range(e.shape[1]):
                    e[i_output,k] =  trainer.constellation[best_sym[k]] - block[i_output,k]
        return e
