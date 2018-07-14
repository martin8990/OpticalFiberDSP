import numpy as np
## Params
import Testing.CaptureLoader as load
import Testing.SimulationLoader as sload
import Testing.Equalizer as eq
import PlotFunctions.MPLMimoPlots as mpl
import matplotlib.pyplot as plt
from qampy import signals, equalisation

# Use Training
#N = 4 *10**4
#sequence,sig = sload.load_8QAM(N)

#Evaluate easiest capture

#N = 5 *10**4
#sequence,sig = load.load_easiest_capture(N)
#eq.equalize(sig,sequence,28e-5,64,20000,True,True,False,equalizer='SBD',showplots=False)

#N = 20 *10**4
#sequence,sig = load.load_easiest_capture(N)
#eq.equalize(sig,sequence,2e-4,64,50000,True,True,False,equalizer='AOLMS')

#N = 20 *10**4
#sequence,sig = load.load_easiest_capture(N)
#eq.equalize(sig,sequence,28e-5,64,50000,True,True,False,equalizer='MRD',showplots =True)


# Harder capture
N = 20 *10**4
sequence,sig = load.load_harder_capture(N)
eq.equalize(sig,sequence,4e-4,32,50000,True,True,False,equalizer='LMS')

#N = 20 *10**4
#sequence,sig = load.load_harder_capture(N)
eq.equalize(sig,sequence,4e-4,32,50000,True,True,False,equalizer='SBD')
#N = 20 *10**4
#sequence,sig = load.load_harder_capture(N)
eq.equalize(sig,sequence,5e-4,32,50000,True,True,False,equalizer='MRD')


# 3Mode Capture
#N = 10 *10**4
#sequence,sig = load.load_3mode_ez_capture(N)
#eq.equalize(sig,sequence,35e-4,32,30000,True,True,False,equalizer = 'LMS',showplots=True)
#eq.equalize(sig,sequence,35e-4,32,90000,True,True,False,equalizer = 'SBD',showplots=True)
#eq.equalize(sig,sequence,35e-4,32,30000,True,True,False,equalizer = 'MRD',showplots= True)

#eq.equalize(sig,sequence,50e-4,32,30000,True,True,False,equalizer = 'SBD')
#eq.equalize(sig,sequence,60e-4,32,30000,True,True,False,equalizer = 'SBD')
#eq.equalize(sig,sequence,70e-4,32,30000,True,True,False,equalizer = 'SBD')
#eq.equalize(sig,sequence,25e-4,32,30000,True,True,False,equalizer = 'MRD')
#eq.equalize(sig,sequence,30e-4,32,30000,True,True,False,equalizer = 'MRD')
#eq.equalize(sig,sequence,40e-4,32,30000,True,True,False,equalizer = 'MRD')


#
#
# 3Mode maximum lb
#print("Stability")
#N = 20 *10**4
#sequence,sig = load.load_3mode_ez_capture(N)
#eq.equalize(sig,sequence,1e-3,400,60000,True,True,False,equalizer = 'LMS')
#eq.equalize(sig,sequence,1e-3,400,60000,True,True,False,equalizer = 'SBD')
#eq.equalize(sig,sequence,1e-3,400,60000,True,True,False,equalizer = 'MRD')
 

## Different algs during training loops