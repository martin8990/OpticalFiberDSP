import numpy as np
## Params
import examples.capture_loader as load
import examples.equalizer as eq
import examples.settings as set # Change your settings in this file

mimo_set = set.MimoSettings()
update_set = set.UpdateSettings()
phaserec_set = set.PhaseRecoverySettings()


nsyms = 50 * 10**4
sequence,sig = load.load_harder_capture(nsyms)

import time
start = time.time()
eq.equalize(sig,sequence,mimo_set,update_set,phaserec_set,showplots=False)
stop = time.time()
print(stop - start)
