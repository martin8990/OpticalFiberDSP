import numpy as np
## Params
import Testing.CaptureLoader as load
import Testing.SimulationLoader as sload
import Testing.Equalizer as eq
import PlotFunctions.MPLMimoPlots as mpl
import matplotlib.pyplot as plt
from qampy import signals
N = 8 *10**4
sequence,sig = sload.load_8QAM(N)
sequence,sig = load.load_easiest_capture(N)
eq.equalize(sig,sequence,True,False)

