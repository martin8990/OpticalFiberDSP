import numpy as np
import mimo.mimo as mimo
import matplotlib.pyplot as plt
import examples.capture_loader as load
## Params
import scipy.io as sio       
import pyqt_mimo.mimoplot as bmp
N = 7 *10**4

def plot_constellation(sig : np.ndarray,title,save_to_file : bool,directory = "") :
    
    alpha = 0.6
    nmodes = sig.shape[0]
    for i_mode in range(nmodes):
        plt.figure(figsize=(5,5))
        plt.scatter(sig[i_mode].real,sig[i_mode].imag,s =0.1,alpha=alpha)
        plt.grid(1)
        plt.ylabel("Quadrature")
        plt.xlabel("InPhase")
        plt.title(title + " Channel " + str(i_mode + 1))
        plt.tight_layout()
        if save_to_file:
            plt.savefig(directory + "//cpl" +str(i_mode)+ ".png")


sequence,sig = load.load_3mode_hard_capture_corr()
corrs = []

for i_mode in range(1):
    corr = np.correlate((sequence[i_mode]),sig[i_mode,1:sequence.shape[1]*2:2],mode = 'full')
    corrs.append(corr)
 

plt.figure()
plt.title('Sequence -> sig')
for i_mode in range(1):
    plt.plot(np.abs(corrs[i_mode]),label = "mode : " + str(i_mode))
    middle = int(len(corrs[i_mode])/2)
    peak = np.abs(corrs[i_mode]).argmax()
    print(peak)
    delta = middle - peak
    print("Mode " + str(i_mode) + " has offset " +str( delta)+ " and maxcor is " + str(corrs[i_mode].max()))
plt.legend()
plot_constellation(sig[0:1,:10000],'test',False)
print(np.unique(sequence,axis = 1))

plt.show()



