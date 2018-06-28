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
from mimo.tap_updater_wide import * 
from mimo.compensator import *
from mimo.phaserecoverer import BlindPhaseSearcher

     
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

# 
# Sample Usage
# 


def equalize_blockwize(block_distr : BlockDistributer,tap_updater : TapUpdater,error_calculator : MimoErrorCalculator,phaserecoverer = None,widely_linear = False):

    nblocks = block_distr.nblocks
    for i_block in range(1,nblocks):
        block_distr.reselect_blocks(i_block)
        if widely_linear:
            compensate_widely(block_distr,tap_updater.H)
        else:
            compensate(block_distr,tap_updater.H)
        if phaserecoverer!=None:
            phaserecoverer.recover_phase(block_distr)
        error_calculator.start_error_calculation(block_distr)

        tap_updater.update_taps(block_distr)
        printProgressBar(i_block,block_distr.nblocks)
    return block_distr.sig_compensated


    

    #
     


if __name__ == '__main__':
    nmodes = 1
    ovsmpl = 1
    nsyms = 1000
    lb = 10

    




