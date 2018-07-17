import numpy as np
from enum import Enum
from examples.options import *


class PhaseRecoverySettings():
   type = PhaseRec.INTERNAL
   lbp = 10 # Blocklength for phaserecovery
   num_testangles = 40


class UpdateSettings():
    mu = 20e-5 # Stepsize
    # error_calculators per section
    # extra training loops ..., training, blind

    # In this case
    # Extra training loop for LMS (15000 syms)
    # then train with SBD (15000 syms)
    # Do the rest of the capture blindly using SBD
    error_calculators = [ECalc.LMS,ECalc.LMS]

    num_trainingsyms = 30000 # also number of training syms in a loop
    phaserec_start = 28000 # Symbol where phaserecovery starts
    update_type = MimoUpdaterType.FREQUENCYDOMAIN

class MimoSettings():
    ovsmpl = 2
    widely_linear = True
    lb = 96 # block length
    name = "LMS,MRD -> MRD" 

