import numpy as np
## Params
import examples.capture_loader as load
import examples.equalizer as eq
from examples.options import *

class PhaseRecoverySettings():
   type = PhaseRec.INTERNAL
   lbp = 15 # Blocklength for phaserecovery
   num_testangles = 40
   search_area = np.pi/2


class UpdateSettings():
    mu = 8e-4 
    # SBD Works better for this capture
    error_calculators = [ECalc.SBD,ECalc.SBD] 

    num_trainingsyms = 50000 
    phaserec_start = 40000 
    update_type = MimoUpdaterType.FREQUENCYDOMAIN

class MimoSettings():
    ovsmpl = 2
    widely_linear = True
    lb = 32 # block length
    name = "Single mode Hard" 

mimo_set = MimoSettings()
update_set = UpdateSettings()
phaserec_set = PhaseRecoverySettings()

nsyms = 7 * 10**4
sequence,sig = load.load_harder_capture(nsyms)
eq.equalize(sig,sequence,mimo_set,update_set,phaserec_set,showplots=True)
