import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,20,20)
y = np.sin(0.1*np.pi*x) + np.sin(0.5*np.pi*x)
plt.figure()
plt.plot(x,y)

Y = np.fft.fft(y)
plt.figure()
plt.subplot(2,1,1)
plt.plot(Y.real)
plt.subplot(2,1,2)
plt.plot(Y.imag)

Y_martin = np.zeros_like(x) + 1j * np.zeros_like(x)
for k in range(0,len(x)):
    cur_sum = 0 + 0j
    for n in range(len(x)):
        print(np.exp(-1j * 2 * np.pi * k * n / len(x)))
        cur_sum += y[n] * np.exp(-1j * 2 * np.pi * k * n / len(x)) 
        
    Y_martin[k] = cur_sum

plt.figure()
plt.subplot(2,1,1)
plt.plot(Y_martin.real)
plt.subplot(2,1,2)
plt.plot(Y_martin.imag)
plt.show()
