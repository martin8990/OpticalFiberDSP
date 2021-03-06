import numpy as np
from examples.options import *

class PhaseRecoverySettings():
   type = PhaseRec.NONE
   lbp = 10 # Blocklength for phaserecovery
   num_testangles = 40
   search_area = np.pi/2


class UpdateSettings():
    mu = 4e-4 # Stepsize
    # error_calculators per section
    # extra training loops ..., training, blind

    # In this case
    # Extra training loop for LMS (15000 syms)
    # then train with SBD (15000 syms)
    # Do the rest of the capture blindly using SBD
    error_calculators = [ECalc.SBD,ECalc.SBD]

    num_trainingsyms = 70000 # also number of training syms in a loop
    phaserec_start = 60000 # Symbol where phaserecovery starts
    update_type = MimoUpdaterType.FREQUENCYDOMAIN

class MimoSettings():
    ovsmpl = 2
    widely_linear = True
    lb = 150 # block length
    name = "LMS,MRD -> MRD" 

