import numpy as np
## Params
import Testing.CaptureLoader as load
import Testing.SimulationLoader as sload
import Testing.Equalizer as eq
N = 7 *10**4

#sequence,sig = sload.load_8QAM(N)
#sequence,sig = load.load_easiest_capture(N)
sequence,sig = load.load_3mode_ez_capture(N)

#sequence[1] = sequence[1].imag + 1j*sequence[1].real
    
print("Loaded signal")
#eq.equalize(sig,sequence)
#eq.equalize_phaserec(sig,sequence)
#eq.equalize_wide_and_phaserec(sig,sequence)
eq.equalize_wide(sig,sequence)