import numpy as np
from pyqt_mimo.mimoplot import MimoFigure

import matplotlib.cm as cm

class ConstellationPlot(MimoFigure):
    
    def _getcolors(self,sig_symsep):
        colors= cm.rainbow(np.linspace(0, 1, len(sig_symsep)))
        return colors[:,:3]
    def __init__(self,sig_symsep,nsyms,name):
        self.sig_symsep = sig_symsep
        self.colors = self._getcolors(sig_symsep)
        self.nsyms= nsyms
        self.name = name
    
    def make_constelation_plot(self,figure,i_bit):
        sig = self.sig_symsep[i_bit]
        col = self.colors[i_bit]
        return figure.plot(sig.real, sig.imag, pen=None, symbol='o', symbolPen=None, symbolSize=4, symbolBrush=(col[0]*255,col[1]*255,col[2]*255, 100))

    def create_figure(self,win):
        self.figure = win.addPlot(title= self.name +" Constellation")
        self.list_plots = []
        for i_bit in range(len(self.sig_symsep)):
            plot = self.make_constelation_plot(self.figure,i_bit)
            self.list_plots.append(plot)
        self.figure.showGrid(x=True, y=True)
        self.figure.setLabel('left', "Quadrature", units='-')
        self.figure.setLabel('bottom', "InPhase", units='-')
        self.figure.setRange(xRange = [-1.2,1.2])
        self.figure.setRange(yRange = [-1.2,1.2])

    def update_region(self, r_min, r_max):
        for i_bit in range(len(self.list_plots)):
            len_arr = len(self.sig_symsep[i_bit])
            bitmin = int(r_min/self.nsyms *len_arr)
            bitmax = int(r_max/self.nsyms * len_arr)
            self.list_plots[i_bit].setData(self.sig_symsep[i_bit].real[bitmin:bitmax],self.sig_symsep[i_bit].imag[bitmin:bitmax])               
 