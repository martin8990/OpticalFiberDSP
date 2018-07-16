import numpy as np
from enum import Enum


class MimoUpdaterType(Enum):
    FREQUENCYDOMAIN = 1
    TIMEDOMAIN = 2

class PhaseRec(Enum):
    INTERNAL = 1
    EXTERNAL = 2
    NONE = 3

class ECalc(Enum):
    LMS = 1
    SBD = 2
    MRD = 3
    CMA = 4


class PhaseRecoverySettings():
   type = PhaseRec.INTERNAL
   lbp = 10 # Blocklength for phaserecovery
   num_testangles = 40


class UpdateSettings():
    mu = 1e-3 # Stepsize
    # error_calculators per section
    # extra training loops ..., training, blind

    # In this case
    # Extra training loop for LMS (15000 syms)
    # then train with SBD (15000 syms)
    # Do the rest of the capture blindly using SBD
    error_calculators = [ECalc.LMS,ECalc.SBD,ECalc.SBD]

    num_trainingsyms = 15000 # also number of training syms in a loop
    phaserec_start = 19000 # Symbol where phaserecovery starts
    update_type = MimoUpdaterType.FREQUENCYDOMAIN

class MimoSettings():
    ovsmpl = 2
    widely_linear = True
    lb = 32 # block length
    name = "LMS,MRD -> MRD" 

