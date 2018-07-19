import numpy as np
from examples.options import *

class PhaseRecoverySettings():
   type = PhaseRec.NONE
   lbp = 10 # Blocklength for phaserecovery
   num_testangles = 40
   search_area = np.pi/2

class UpdateSettings():
    mu = 6e-4 
    error_calculators = [ECalc.LMS,ECalc.LMS]

    num_trainingsyms = 50000 # also number of training syms in a loop
    phaserec_start = 40000 # Symbol where phaserecovery starts
    update_type = MimoUpdaterType.FREQUENCYDOMAIN

class MimoSettings():
    ovsmpl = 2
    widely_linear = True
    lb = 32 # block length
    name = "Single mode Hard" 

