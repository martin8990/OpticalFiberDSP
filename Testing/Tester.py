import numpy as np
## Params
import Testing.CaptureLoader as load
import Testing.SimulationLoader as sload
import Testing.Equalizer as eq
N = 5 *10**4

#sequence,sig = sload.load_8QAM()
sequence,sig= load.load_easiest_capture()
sig = sig[:,:N*2]
sequence = sequence[:,:N]
nmodes = sig.shape[0]
print("Loaded signal")
#eq.equalize(sig,sequence)
eq.equalize_phaserec(sig,sequence)