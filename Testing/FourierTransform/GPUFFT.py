import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import cmath as mth
import timeit
N = 1024

@cuda.jit
def cudaFFT(arr,out,len_arr):
    k = cuda.threadIdx.x
    cur_sum = 0
    for n in range(len_arr):
        cur_sum += arr[n] * mth.exp(-1j * 2 * mth.pi * k * n / len_arr) 
    out[k] = cur_sum

x = np.linspace(0,20,N)
y = np.sin(0.1*np.pi*x) + np.sin(0.5*np.pi*x) + 0j

y_gpu = cuda.to_device(y)

a = cudaFFT[1,N]
Y = cuda.device_array_like(y)
for k in range(8):
    start = timeit.timeit()
    cudaFFT[1,N](y_gpu,Y,N)
    end = timeit.timeit()
    print(end - start)



Y = Y.copy_to_host()
plt.figure()
plt.subplot(2,1,1)
plt.plot(Y.real)
plt.subplot(2,1,2)
plt.plot(Y.imag)

Y = np.fft.fft(y)
plt.figure()
plt.subplot(2,1,1)
plt.plot(Y.real)
plt.subplot(2,1,2)
plt.plot(Y.imag)

