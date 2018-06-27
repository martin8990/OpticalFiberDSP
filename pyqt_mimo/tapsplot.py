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
            fig = win.addPlot(title= "Mode : " + str(k)+  " -> " +  self.name )
            real_taps, imag_taps = self._plot_taps(fig,self.taps[k])
            self.list_real_taps.append(real_taps)
            self.list_imag_taps.append(imag_taps)

    def update_region(self, r_min, r_max):
        for i_input in range(len(self.list_imag_taps)):
            t_tap = r_min / self.nsyms
            i_tap = min(max([0,int(t_tap * self.ntaps)]),self.taps.shape[1]-1)
            self.list_real_taps[i_input].setData(self.taps.real[i_input,i_tap])
            self.list_imag_taps[i_input].setData(self.taps.imag[i_input,i_tap])

