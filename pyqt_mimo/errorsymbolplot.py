import numpy as np
from pyqt_mimo.mimoplot import MimoFigure

import matplotlib.cm as cm
import mimo.mimo as mimo

class ErrorSymbolPlot(MimoFigure):
    
    def _getcolors(self,sig_symsep):
        colors= cm.rainbow(np.linspace(0, 1, len(sig_symsep)))
        return colors[:,:3]

    def sort_sig_per_sym(self,sig,trainer,i_mode):
        sig_sym = []
        ctl = trainer.constellation
        symids= trainer.symids[i_mode]
     
        sig_sym_mode = []
        for i_bit in range(len(ctl)):
            sig_sym_mode.append([])    
        for k in range(sig.shape[0]):
            my_answer = symids[k]
            sig_sym_mode[my_answer].append(sig[k])
        sig_sym_mode_arr = []
        for i_bit in range(len(ctl)):
            sig_sym_mode_arr.append(np.asarray(sig_sym_mode[i_bit]))
        return sig_sym_mode_arr

    def __init__(self,sig,i_mode,trainer : mimo.Trainer,nsyms,name):
        
        # Get ids where decision != symids
        mysig = sig.copy()
        correct_ids = np.argwhere(trainer.decisions[i_mode]-trainer.symids[i_mode]==0)
        for k in correct_ids:
            mysig[k] = 0

        self.sig_symsep = self.sort_sig_per_sym(mysig,trainer,i_mode)
        self.colors = self._getcolors(self.sig_symsep)
        self.nsyms= nsyms
        self.name = name
        self.constellation = trainer.constellation
       
    
    def make_constelation_plot(self,figure,i_bit):
        sig = self.sig_symsep[i_bit]
        col = self.colors[i_bit]
        ctl = np.asarray(self.constellation[i_bit])
        return figure.plot(sig.real, sig.imag, pen=None, symbol='o', symbolPen=None, symbolSize=10, symbolBrush=(col[0]*255,col[1]*255,col[2]*255, 255))

    def create_figure(self,win):
        self.figure = win.addPlot(title= self.name +" Errors")
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
            mydata = self.sig_symsep[i_bit][bitmin:bitmax]
            mydata = mydata[mydata!=0]
            self.list_plots[i_bit].setData(mydata.real,mydata.imag)               
 