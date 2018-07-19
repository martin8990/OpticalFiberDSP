import numpy as np
## Params
import examples.capture_loader as load
import examples.equalizer as eq
from examples.options import *

class PhaseRecoverySettings():
   type = PhaseRec.NONE
   lbp = 10 # Blocklength for phaserecovery
   num_testangles = 40
   search_area = np.pi/2
   
class UpdateSettings():
    mu = 4e-4 # Stepsize
    error_calculators = [ECalc.SBD,ECalc.SBD]

    num_trainingsyms = 70000 # Requires a lot of training
    phaserec_start = 60000 
    update_type = MimoUpdaterType.FREQUENCYDOMAIN

class MimoSettings():
    ovsmpl = 2
    widely_linear = True
    #lb = 150 # block length
    lb = 250
    name = "3 Mode Hard" 

mimo_set = MimoSettings()
update_set = UpdateSettings()
phaserec_set = PhaseRecoverySettings()

nsyms = 13 * 10**4
sequence,sig = load.load_3mode_hard_capture(nsyms)
eq.equalize(sig,sequence,mimo_set,update_set,phaserec_set,showplots=True)
