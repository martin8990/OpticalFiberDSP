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
from mimo.blockdistributer import BlockDistributer
from mimo.trainer import *

from mimo.error_calculator import *
from mimo.tap_updater import *

from mimo.compensator import *
from mimo.phaserecoverer import BlindPhaseSearcher
from utility.cmd import printProgressBar
import time
# 
# Sample Usage
# 


def equalize_blockwize(block_distr : BlockDistributer,tap_updater : TapUpdater,phaserecoverer = None,widely_linear = False):
    trainer = block_distr.trainer
    nblocks = block_distr.nblocks
    for i_block in range(1,nblocks):
        block_distr.reselect_blocks(i_block)
        compensate(block_distr,tap_updater.H)
       
        if phaserecoverer!=None:
            phaserecoverer.recover_phase(block_distr,trainer)
        
        error_calculator = trainer.get_error_calculator(block_distr)
        error_calculator.start_error_calculation(block_distr,trainer)

        tap_updater.update_taps(block_distr)
        printProgressBar(i_block,block_distr.nblocks)
    return block_distr.sig_compensated

def equalize_blockwize_timed(block_distr : BlockDistributer,tap_updater : TapUpdater,phaserecoverer = None,widely_linear = False):
    trainer = block_distr.trainer
    nblocks = block_distr.nblocks
    times = np.zeros(5)
    for i_block in range(1,nblocks):
        start = time.time()
        block_distr.reselect_blocks(i_block)
        times[0]+=time.time()-start
       
        start = time.time()
        compensate(block_distr,tap_updater.H)
        times[1]+=time.time()-start
       
        start = time.time()
        if phaserecoverer!=None:
            phaserecoverer.recover_phase(block_distr,trainer)
        times[2]+=time.time()-start
       
        start = time.time()
        error_calculator = trainer.get_error_calculator(block_distr)
        error_calculator.start_error_calculation(block_distr,trainer)
        times[3]+=time.time()-start

        start = time.time()
        tap_updater.update_taps(block_distr)
        times[4]+=time.time()-start
        print(times)
    return block_distr.sig_compensated
    

    #
     


if __name__ == '__main__':
    nmodes = 1
    ovsmpl = 1
    nsyms = 1000
    lb = 10

    




