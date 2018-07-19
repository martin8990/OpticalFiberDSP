import numpy as np
from pyqt_mimo.mimoplot import MimoFigure
from PyQt5 import QtCore 
class TapsPlot(MimoFigure):
    def __init__(self,taps,nsyms,name):
        self.taps = taps
        self.nsyms = nsyms
        self.ntaps = taps.shape[1]
        self.name = name

    def _plot_taps(self,figure,taps):
        figure.addLegend()
        figure.showGrid(x=True, y=True)
        figure.setLabel('left', "weight", units='-')
        figure.setLabel('bottom', "tap", units='-')
        return figure.plot(taps.real[-1],pen = (0,255,0,255),name='real'), figure.plot(taps.imag[-1],pen = (255,0,0,255),name='imag')

    def create_figure(self, win):
        self.list_real_taps = []
        self.list_imag_taps = [] 
        for k in range(self.taps.shape[0]):
            if self.taps.shape[0] > 1:
                fig = win.addPlot(title= "Mode : " + str(k)+  " -> " +  self.name )
            else :
                fig = win.addPlot(title= self.name )
            real_taps, imag_taps = self._plot_taps(fig,self.taps[k])
            self.list_real_taps.append(real_taps)
            self.list_imag_taps.append(imag_taps)

    def update_region(self, r_min, r_max):
        for i_input in range(len(self.list_imag_taps)):
            t_tap = r_min / self.nsyms
            i_tap = min(max([0,int(t_tap * self.ntaps)]),self.taps.shape[1]-1)
            self.list_real_taps[i_input].setData(self.taps.real[i_input,i_tap])
            self.list_imag_taps[i_input].setData(self.taps.imag[i_input,i_tap])





class TapsPlotMerged(MimoFigure):
    def __init__(self,taps,nsyms,i_mode,name):
        self.taps = taps[:,0]
        self.nsyms = nsyms
        self.taps_conj = taps[:,1]
        self.ntaps = taps.shape[2]
        self.nmodes = taps.shape[0]
        self.name = name
        self.i_mode = i_mode
        self.i_input = i_mode

    def create_figure(self, win):
        i_input = self.i_mode
        fig = win.addPlot(title= "Mode : " + str(i_input)+  " -> " +  self.name )
        self.lgnd = fig.addLegend()
        fig.showGrid(x=True, y=True)
        fig.setLabel('left', "weight", units='-')
        fig.setLabel('bottom', "tap", units='-')
        fig.setYRange(-0.5,0.5,update=False)
        fig.keyPressEvent = self.on_key_pressed
        self.fig = fig
        self.real_plt = fig.plot(self.taps[i_input,-1].real,pen = ('g'),name='real')
        self.imag_plt = fig.plot(self.taps.imag[i_input,-1],pen = ('r'),name='imag')
        self.real_conj_plt = fig.plot(self.taps_conj.real[i_input,-1],pen = ('b'),name='real_conj')
        self.imag_conj_plt = fig.plot(self.taps_conj.imag[i_input,-1],pen = ('w'),name='imag_conj')
         

    def on_key_pressed(self,event):
        if event.key() == QtCore.Qt.Key_Space:
            self.i_input+=1
            
            if self.i_input == self.nmodes:
                self.i_input = 0
                
            self.fig.setTitle( "Mode : " + str(self.i_input)+  " -> " +  self.name)
            self.update_region(self.r_min,self.r_max) 
            
        


    def update_region(self, r_min, r_max):
        i_input = self.i_input        
        t_tap = r_min / self.nsyms
        i_block = min(max([0,int(t_tap * self.ntaps)]),self.taps.shape[1]-1)
        self.real_plt.setData(self.taps.real[i_input,i_block])
        self.imag_plt.setData(self.taps.imag[i_input,i_block])
        self.real_conj_plt.setData(self.taps_conj.real[i_input,i_block])
        self.imag_conj_plt.setData(self.taps_conj.imag[i_input,i_block])
        self.r_min = r_min
        self.r_max = r_max
