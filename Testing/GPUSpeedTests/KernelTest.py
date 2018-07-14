import scipy.io as sio
import numpy as np
import numba.cuda as cuda
import Testing.FourierTransform.GPUFFT as GFFT
import pyculib.fft as fft

@cuda.jit
def DoSomething(arr):
    x = cuda.threadIdx.x
    y = cuda.threadIdx.y
    z = cuda.threadIdx.z
    w = cuda.blockDim.x
    arr[x,y,z,w] = x + y + z + w

@cuda.jit
def DoLoopGpu(arr,nloops):
    x = cuda.threadIdx.x
    y = cuda.threadIdx.y
    z = cuda.threadIdx.z
    w = cuda.blockIdx.x
    
    for k in range(nloops):
        arr[x,y,z,w] = k+1

@cuda.jit
def DoLoopGpu(arr,nloops):
    x = cuda.threadIdx.x
    y = cuda.threadIdx.y
    z = cuda.threadIdx.z
    w = cuda.blockIdx.x
    
    for k in range(nloops):
        arr[x,y,z,w] = k+1

@cuda.jit
def DoLoopGpuWSync(arr,nloops):
    x = cuda.threadIdx.x
    y = cuda.threadIdx.y
    z = cuda.threadIdx.z
    w = cuda.blockIdx.x
    
    for k in range(nloops):
        arr[x,y,z,w] = k+1
        cuda.syncthreads()

shp = (2,2,2,64)
shp_thread = (2,2,64)
shp_block = (2)

My_arr = cuda.device_array(shp)
def DoLoopCPU(My_arr,nloops):
    for k in range(nloops):
        DoSomething[shp_block,(shp_thread)](My_arr)
import time


DoLoopCPU(My_arr,1)
DoSomething[10,(10,10,10)](My_arr)
print("Initialized")

nloops = 30000
start = time.time()
DoLoopCPU(My_arr,30000)
end = time.time()
print("CPU time per sync : " + str((end-start)/nloops))


DoLoopGpuWSync[shp_block,(shp_thread)](My_arr,1)
print("Initialized")
nloops = 300000000
start = time.time()
DoLoopGpuWSync[shp_block,(shp_thread)](My_arr,nloops)
end = time.time()

print("GPU time per sync : " + str((end-start)/nloops))

#arr_cpu = My_arr.copy_to_host()
#print(arr_cpu[1,1,1,1])








