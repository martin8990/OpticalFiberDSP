import numpy as np
import matplotlib.pyplot as plt
## Params

A = np.random.rand(100) + np.random.rand(100) * 1j
normal = np.fft.fft(A)
conj = np.fft.fft(np.conj(A))
normalconj = np.conj(normal)
conj - normalconj