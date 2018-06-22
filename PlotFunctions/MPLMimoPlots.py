import numpy as np
import matplotlib.pyplot as plt

def plot_constellation_1mode(sig,title,k_start,k_stop,mode_id) :
    
    plt.figure()
    plt.scatter(sig[mode_id,k_start:k_stop].real,sig[mode_id,k_start:k_stop].imag,s =0.1,alpha=0.5)
    plt.grid(1)
    plt.ylabel("Quadrature")
    plt.xlabel("InPhase")
    plt.title(title)

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


def plot_constellation_bitsep(bit_sigs,constellation,title,save_to_file : bool,directory = "") :
    
    alpha = 0.6
    nmodes = len(bit_sigs)
    for i_mode in range(nmodes):
        plt.figure(figsize=(5,5))
        for i_bit in range(len(constellation)):
            label = "{:.2f}".format( constellation[i_bit] )
            plt.scatter(bit_sigs[i_mode][i_bit].real,bit_sigs[i_mode][i_bit].imag,s =0.1,alpha=alpha,label = label)            
        plt.grid(1)
        plt.ylabel("Quadrature")
        plt.xlabel("InPhase")
        plt.title(title + " Channel " + str(i_mode + 1))
        plt.tight_layout()
        lgnd = plt.legend()

        for i_bit in range(len(constellation)):
            lgnd.legendHandles[i_bit]._sizes = [30]
        if save_to_file:
            plt.savefig(directory + "//cpl" +str(i_mode)+ ".png")




 
def plot_error(err:np.ndarray,title,save_to_file : bool,directory):
    nmodes = len(err)
    for i_mode in range(nmodes):
        plt.figure()
        plt.plot(err[0])
        plt.grid(1)
        plt.ylabel("Error (-)")
        plt.xlabel("Symbol (-)")
        plt.tight_layout()
        plt.title(title + " Channel " + str(i_mode + 1))
        if save_to_file:
            plt.savefig(directory + "\epl"+str(i_mode)+ ".png")


def plot_taps(taps:np.ndarray,save_to_file : bool,directory = ""):
    plt.figure(figsize=(7,7))
    nmodes = len(taps[:,0,0])
    cnt = 0
    for x in range(nmodes):
        for y in range(nmodes):
            cnt+=1
            plt.subplot(nmodes,nmodes,cnt)
            plt.plot(taps[x,y].real,label = "real")
            plt.plot(taps[x,y].imag,label = "imag")
            plt.grid(1)
            plt.ylabel("weight (-)")
            plt.xlabel("tap (-)")
            plt.legend()
            
    plt.tight_layout()
    if save_to_file:        
        plt.savefig(directory + "\_tpl.png")
    