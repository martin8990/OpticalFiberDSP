import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import cmath as mth
import timeit
import pyculib.fft as fft
import time 
import numba 

lb = 64
sizediv = 8

nmodes = 1
ovsmpl = 1
ovconj = 1

data = np.ones((lb),dtype = np.complex128)
nblocks = 10000


shp_pyc = (lb)
plan = fft.FFTPlan((lb,),np.complex128,np.complex128,1)

arr_pyculib = cuda.to_device(data.reshape(shp_pyc))
arr_out_pyculib = cuda.device_array_like(arr_pyculib)
start = time.time()
for i_block in range(int(nblocks)):
    plan.forward(arr_pyculib,arr_out_pyculib)


print(arr_out_pyculib[0])
end = time.time()
print("Time pyculib = " + str(end-start))

arr_Martin = cuda.to_device(data)
arr_out_martin = cuda.device_array_like(arr_Martin)



@cuda.jit
def cuda_fft_1d_mass_optimized(arr,out,len_arr,nloops):
    k = cuda.blockIdx.x
    k2 = cuda.threadIdx.x
    precalc = -1j * 2 * mth.pi * k /len_arr
    presum = cuda.shared.array(shape=(sizediv), dtype=numba.complex64)
    for i_loop in range(nloops):
        cur_sum = 0
        for n in range(sizediv):
            cur_sum += arr[k] * mth.exp(precalc* (n + k2 * sizediv))
        presum[k2] = cur_sum
        cuda.syncthreads()
        if k2==0:
            cur_sum = 0
            for n in range(sizediv):
                cur_sum += presum[n]
            out[k] = cur_sum

@cuda.jit
def cuda_fft_1d_mass_optimized2(arr,out,nloops):
    x = cuda.threadIdx.x
    y = cuda.threadIdx.y
    precalc = -1j * 2 * mth.pi * y /len_arr
    presum = cuda.shared.array(shape=(sizediv,sizediv), dtype=numba.complex64)
    #for i_loop in range(nloops):
    #    cur_sum = 0
    #    for n in range(sizediv):
    #        cur_sum += mth.exp(precalc* (n + x * sizediv))
    #    presum[x] = cur_sum
    #    cuda.syncthreads()
    #    if k2==0:
    #        for n in range(sizediv):
    #            cur_sum += presum[n]
    #        for k in range(sizediv):
    #            out[k + sizediv * x] = cur_sum * arr[k + sizediv * x]


        #cuda.s yncthreads()

@cuda.jit
def cudaFFt1dMass(arr,out,len_arr,nloops):
    k = cuda.threadIdx.x
    precalc = -1j * 2 * mth.pi * k /len_arr
    for i_loop in range(nloops):
        cur_sum = 0
        for n in range(len_arr):
            cur_sum += arr[k] * mth.exp(precalc* n) 
        out[k] = cur_sum
        #cuda.syncthreads()



cuda_fft_1d_mass_optimized2[(32,32),(32,32)](arr_Martin,arr_out_martin,lb,1)
print("Init")

start = time.time()
cuda_fft_1d_mass_optimized2[(32,32),(32,32)](arr_Martin,arr_out_martin,lb,int(nblocks))
print(arr_out_martin.copy_to_host()[0])
end = time.time()
print("Time Martin = " + str(end-start))

start = time.time()

for k in range(nblocks):
    arr_Martin = arr_Martin.reshape(2,lb/2)
    arr_Martin = arr_Martin.reshape(lb)
end = time.time()
print("Time Reshape = " + str(end-start))


