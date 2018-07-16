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
    
        self.nmodes = block_distr.nmodes
        self.comp_pr = True

    def start_error_calculation(self,block_distr : BlockDistributer,trainer : Trainer):
        block = block_distr.block_compensated
        i_block = block_distr.i_block
        err = self.calculate_error(block,trainer)
        if self.comp_pr:
            err = err*np.exp(1.j*-block_distr.phases)
        block_distr.insert_block_error(err)
        self.error[:,i_block*block_distr.lb:(i_block+1)*block_distr.lb] = err
    
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

class TrainedLMSAmpOnly(MimoErrorCalculator):
    def __init__(self,block_distr : BlockDistributer):
        super().__init__(block_distr)
      
        

    # Insert one block for each input
    def calculate_error(self,block,trainer : Trainer):
        if trainer.in_training:
            sym_best = trainer.block_sequence
            e = sym_best - block
        else :
            e = np.zeros_like(block)
            for i_output in range(block.shape[0]):
                best_sym = np.argmin(np.abs( trainer.constellation-block[i_output,:,np.newaxis]),axis=1) 
                for k in range(e.shape[1]):
                    sym = block[i_output,k] 
                    sym_best = trainer.constellation[best_sym[k]]
                    e[i_output,k] = (sym_best * np.conj(sym_best)-sym * np.conj(sym))*sym_best
        return e


class TrainedSBD(MimoErrorCalculator):

    # Insert one block for each input
    def calc_e_sbs(self,best_syms,block):
        ar = best_syms.real
        yr = block.real
        ai = best_syms.imag
        yi = block.imag
        e = np.abs(ar)*(ar - yr) + 1j * np.abs(ai) *(ai-yi)
        return e
    def calculate_error(self,block,trainer : Trainer):
        
        if trainer.in_training:
            return self.calc_e_sbs(trainer.block_sequence,block)

        else :
            best_decision = np.argmin(np.abs( trainer.constellation-block[:,:,np.newaxis]),axis=2) 
            best_sym = np.zeros_like(block)
            for i_output in range(best_sym.shape[0]):
                for k in range(best_sym.shape[1]):
                    best_sym[i_output,k] =  trainer.constellation[best_decision[i_output,k]]
                               
            return self.calc_e_sbs(best_sym,block)


class TrainedMRD(MimoErrorCalculator):
    # Filho, M., Silva, M. T. M., & Miranda, M. D. (2008). A FAMILY OF ALGORITHMS FOR BLIND EQUALIZATION OF QAM SIGNALS. In 2011 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 6–9).

    # Insert one block for each input
    def calc_e_mrd(self,best_syms,block):
        ar = best_syms.real
        yr = block.real
        ai = best_syms.imag
        yi = block.imag
        e = yr*(ar**2 - yr**2) + 1j * yi *(ai**2-yi**2)
        return e
    def calculate_error(self,block,trainer : Trainer):
        if trainer.in_training:
            return self.calc_e_mrd(trainer.block_sequence,block)
        else :
            best_decision = np.argmin(np.abs( trainer.constellation-block[:,:,np.newaxis]),axis=2) 
            best_sym = np.zeros_like(block)
            for i_output in range(best_sym.shape[0]):
                for k in range(best_sym.shape[1]):
                    best_sym[i_output,k] =  trainer.constellation[best_decision[i_output,k]]
                               
            return self.calc_e_mrd(best_sym,block)        