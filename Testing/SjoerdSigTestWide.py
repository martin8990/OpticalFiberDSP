import numpy as np
## Params
import Testing.CaptureLoader as load       
import Testing.Equalizer as eq
N = 5 *10**4

sequence,sig= load.load_easiest_capture()
sig = sig[:,:N*2]
sequence = sequence[:,:N]
nmodes = sig.shape[0]

eq.equalize_wide(sig,sequence)
