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

def load_3mode_ez_capture():
    capture = sio.loadmat('3M_5.46E-5_21taps.mat', squeeze_me=True)
    sequence = capture['Sequence'][()].astype(np.complex128)
    sequence[0] = np.roll(sequence[0],-11350)
    sequence[1] = np.roll(sequence[1],-6301)
    sequence[2] = np.roll(sequence[2],-9624)
    sequence[3] = np.roll(sequence[3],-32658)
    sequence[4] = np.roll(sequence[4],-13081)
    sequence[5] = np.roll(sequence[5],31149)

    sequence[1] = sequence[1].imag + 1j*sequence[1].real
    #sequence[3] = sequence[1].imag + 1j*sequence[1].real
    #sequence[5] = sequence[1].imag + 1j*sequence[1].real

    sig = capture['Input'][()].astype(np.complex128)
    sig = sig*4
    return sequence,sig