#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.colors import LinearSegmentedColormap

#def plot_constellation_heatmap(constelation : np.ndarray,BER,resolution,save_to_file : bool,directory = ""):
#    plt.figure(figsize=(8,8))
#    nmodes = len(constelation[:,0])
#    nmult = 10
#    colors = [0,0,0]
#    for k in range(nmult):
#        colors = colors.append([0,0,k * 1/nmult])
#    for k in range(nmult):
#        colors = colors.append([k*1/nmult,0,1-k*1/nmult])


#    n_bin = 100  # Discretizes the interpolation into bins
#    cm = LinearSegmentedColormap.from_list('my_list', colors, N=n_bin)
#    for i_mode in range(nmodes):        
#        ZZ = np.zeros(resolution * resolution).reshape(resolution,resolution)
#        for k in range(len(constelation[i_mode])):
#            x = int((constelation[i_mode,k].real+1)/2*resolution)  
#            y = int((constelation[i_mode,k].imag+1)/2*resolution)

#            if x<resolution and y<resolution and x>=0 and y >= 0:
#                ZZ[x,y]+=10000
#        nlabels = 11
#        ticks = np.linspace(-1,1,nlabels)
#        xtickLabels = []
#        ytickLabels = []

#        for tick in ticks:
#            xtickLabels.append("%.1f" % tick)
#            ytickLabels.append("%.1f" % -tick)
#        plt.subplot(nmodes,1,i_mode+1)    
        
#        tickPositions = np.linspace(0,resolution,nlabels)
#        plt.imshow(ZZ,cmap = cm)
#        plt.xlabel("InPhase(-)")
#        plt.ylabel("Quadrature(-)")
#        plt.xticks(tickPositions,xtickLabels)
#        plt.yticks(tickPositions,xtickLabels)
#        plt.text(230, 100, "BER : " + "%.4f" %BER[i_mode],fontsize=13)
#    if save_to_file:
#       plt.savefig(directory +  '/heatmap.pdf')
#    else:
#       plt.show()
#if __name__ == '__main__':
#    N = 4 * 10 ** 4
#    x = np.arange(0,N,1)

    
#    real = np.random.rand(N*2)*2-1
#    imag = np.random.rand(N*2)*2-1
#    Constellation = (real + 1j * imag).reshape(2,N)
#    BER = [0.00223,0.005]
#    plot_constellation_heatmap(Constellation,BER,200,False)
#    plt.show()