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
from mimo.error_calculator import *
from mimo.tap_updater import *
from mimo.compensator import *
from mimo.phaserecoverer import BlindPhaseSearcher

     
def equalize_blockwize(block_distr : BlockDistributer,tap_updater : TapUpdater,error_calculator : MimoErrorCalculator):

    nblocks = block_distr.nblocks
    for i_block in range(1,nblocks):
        block_distr.reselect_blocks(i_block)
        compensate(block_distr,tap_updater.H)
        error_calculator.start_error_calculation(block_distr)
        tap_updater.update_taps(block_distr)
    return block_distr.sig_compensated

def equalize_blockwize_with_phaserec(block_distr : BlockDistributer,tap_updater : TapUpdater,error_calculator : MimoErrorCalculator, phaserecoverer : BlindPhaseSearcher):

    nblocks = block_distr.nblocks
    for i_block in range(1,nblocks):
        block_distr.reselect_blocks(i_block)
        compensate(block_distr,tap_updater.H)
        phaserecoverer.recover_phase(block_distr)
        error_calculator.start_error_calculation(block_distr)
        tap_updater.update_taps(block_distr)
    return block_distr.sig_compensated
    

    #
     


if __name__ == '__main__':
    nmodes = 1
    ovsmpl = 1
    nsyms = 1000
    lb = 10

    




