import scipy.io as sio
import numpy as np
def load_easiest_capture(N):
    capture = sio.loadmat('QSM_3.10E-6_51taps.mat', squeeze_me=True)
    sequence = capture['Sequence'][()].astype(np.complex128)
    sequence = np.roll(sequence,24676,axis =1)
    
    while N > sequence.shape[1]:
        sequence = np.append(sequence,sequence,axis=1)
        sequence = sequence[:,:N]
    sequence[1] = sequence[1].imag + 1j*sequence[1].real
    
    sig = capture['Input'][()].astype(np.complex128)
    sig = sig*20
    sig = sig[:,:N*2]
    
    return sequence,sig

def load_harder_capture(N):
    capture = sio.loadmat('QSM_3.25E-3_51taps.mat', squeeze_me=True)
    sequence = capture['Sequence'][()].astype(np.complex128)
    sequence = np.roll(sequence,-4000+385,axis =1)
    while N > sequence.shape[1]:
        sequence = np.append(sequence,sequence,axis=1)
        sequence = sequence[:,:N]

    sequence[1] = sequence[1].imag + 1j*sequence[1].real
    
    sig = capture['Input'][()].astype(np.complex128)
    sig = sig*4
    sig = sig[:,:N*2]
    return sequence,sig

def load_3mode_ez_capture(N):
    capture = sio.loadmat('3M_5.46E-5_21taps.mat', squeeze_me=True)
    sequence = capture['Sequence'][()].astype(np.complex128)
    sequence = np.roll(sequence,-11350,axis = 1)
    
    while N > sequence.shape[1]:
        sequence = np.append(sequence,sequence,axis=1)
        sequence = sequence[:,:N]


    sig = capture['Input'][()].astype(np.complex128)
    sig = sig*4
    sig = sig[:,:N*2]
    return sequence,sig

def load_3mode_hard_capture(N):
    capture = sio.loadmat('3M_9.50E-3_701taps.mat', squeeze_me=True)
    sequence = capture['Sequence'][()].astype(np.complex128)
    sequence = np.roll(sequence,-11350,axis = 1)
    
    while N > sequence.shape[1]:
        sequence = np.append(sequence,sequence,axis=1)
        sequence = sequence[:,:N]


    sig = capture['Input'][()].astype(np.complex128)
    sig = sig*4
    sig = sig[:,:N*2]
    return sequence,sig