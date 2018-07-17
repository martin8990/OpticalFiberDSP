import numpy as np
from enum import Enum
from examples.options import *


class PhaseRecoverySettings():
   type = PhaseRec.INTERNAL
   lbp = 10 # Blocklength for phaserecovery
   num_testangles = 20


class UpdateSettings():
    mu = 20e-5 # Stepsize
    error_calculators = [ECalc.LMS,ECalc.LMS]

    num_trainingsyms = 30000 # also number of training syms in a loop
    phaserec_start = 28000 # Symbol where phaserecovery starts
    update_type = MimoUpdaterType.FREQUENCYDOMAIN

class MimoSettings():
    ovsmpl = 2
    widely_linear = True
    lb = 128 # block length
    name = "64 Qam Settings" 

