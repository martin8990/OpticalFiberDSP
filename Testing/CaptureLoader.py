import scipy.io as sio
import numpy as np
def load_easiest_capture():
    capture = sio.loadmat('QSM_3.10E-6_51taps.mat', squeeze_me=True)
    sequence = capture['Sequence'][()].astype(np.complex128)
    sequence = np.roll(sequence,24676,axis =1)
    sequence[1] = sequence[1].imag + 1j*sequence[1].real
    sig = capture['Input'][()].astype(np.complex128)
    sig = sig*20
    return sequence,sig

def load_harder_capture():
    capture = sio.loadmat('QSM_3.25E-3_51taps.mat', squeeze_me=True)
    sequence = capture['Sequence'][()].astype(np.complex128)
    sequence = np.roll(sequence,-4000+385,axis =1)
    sequence[1] = sequence[1].imag + 1j*sequence[1].real
    sig = capture['Input'][()].astype(np.complex128)
    sig = sig*4
    return sequence,sig