import numpy as np
## Params
import examples.capture_loader as load
import examples.equalizer as eq
from examples.options import *
import matplotlib.pyplot as plt


class PhaseRecoverySettings():
   type = PhaseRec.INTERNAL
   lbp = 10 # Blocklength for phaserecovery
   num_testangles = 20
   search_area = np.pi/2


class UpdateSettings():
    mu = 20e-5 # Stepsize
    error_calculators = [ECalc.LMS,ECalc.LMS]

    num_trainingsyms = 40000 # Phase slips somehow occur
    phaserec_start = 38000 
    update_type = MimoUpdaterType.FREQUENCYDOMAIN

class MimoSettings():
    ovsmpl = 2
    widely_linear = True
    lb = 128 # block length
    name = "64 Qam Phase slip" 

mimo_set = MimoSettings()
update_set = UpdateSettings()
phaserec_set = PhaseRecoverySettings()

nsyms = 7 * 10**4
sequence,sig = load.load_64Qam_best(nsyms)
sig_eq = eq.equalize(sig,sequence,mimo_set,update_set,phaserec_set,showplots=False)
plt.figure()
plt.scatter(sig_eq[0,35000:60000].real,sig_eq[0,35000:60000].imag)
plt.show()


