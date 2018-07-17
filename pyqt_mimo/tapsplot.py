import numpy as np
from pyqt_mimo.mimoplot import MimoFigure

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
    def __init__(self,taps,taps_conj,nsyms,name):
        self.taps = taps
        self.nsyms = nsyms
        self.taps_conj = taps_conj
        self.ntaps = taps.shape[1]
        self.name = name

    def create_figure(self, win):
        self.list_real_taps = []
        self.list_imag_taps = [] 
        self.list_real_taps_conj = []
        self.list_imag_taps_conj = [] 

        for k in range(self.taps.shape[0]):
            if self.taps.shape[0] > 1:
                fig = win.addPlot(title= "Mode : " + str(k)+  " -> " +  self.name )
            else :
                fig = win.addPlot(title= self.name )
            #fig.addLegend()
            fig.showGrid(x=True, y=True)
            fig.setLabel('left', "weight", units='-')
            fig.setLabel('bottom', "tap", units='-')
            real_taps = fig.plot(self.taps[k,-1].real,pen = ('g'),name='real')
            imag_taps = fig.plot(self.taps.imag[k,-1],pen = ('r'),name='imag')
            real_taps_conj = fig.plot(self.taps_conj.real[k,-1],pen = ('b'),name='real_conj')
            imag_taps_conj = fig.plot(self.taps_conj.imag[k,-1],pen = ('w'),name='imag_conj')
           
            #self.list_real_taps.append(real_taps)
            #self.list_imag_taps.append(imag_taps)
            #self.list_real_taps_conj.append(real_taps_conj)
            #self.list_imag_taps_conj.append(imag_taps_conj)

    def update_region(self, r_min, r_max):
        for i_input in range(len(self.list_imag_taps)):
            t_tap = r_min / self.nsyms
            i_tap = min(max([0,int(t_tap * self.ntaps)]),self.taps.shape[1]-1)
            self.list_real_taps[i_input].setData(self.taps.real[i_input,i_tap])
            self.list_imag_taps[i_input].setData(self.taps.imag[i_input,i_tap])
            self.list_real_taps_conj[i_input].setData(self.taps_conj.real[i_input,i_tap])
            self.list_imag_taps_conj[i_input].setData(self.taps_conj.imag[i_input,i_tap])

