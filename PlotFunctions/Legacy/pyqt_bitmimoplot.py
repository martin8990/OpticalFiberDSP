

from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import matplotlib.cm as cm

class BitMimoPlotRequest : 
    def __init__(self,error : np.array,bit_sigs,taps : np.array,phase,slipups,slipdowns,name : str):
        self.error = error
        self.bit_sigs = bit_sigs
        self.taps = taps
        self.phase = phase
        self.slipups = slipups
        self.slipdowns = slipdowns
        self.name = name
        self.colors = self._getcolors(bit_sigs)
    def _getcolors(self,bit_sigs):
        colors= cm.rainbow(np.linspace(0, 1, len(bit_sigs)))
        return colors[:,:3]


def make_constelation_plot(figure,signal,col):
        figure.showGrid(x=True, y=True)
        figure.setLabel('left', "Quadrature", units='-')
        figure.setLabel('bottom', "InPhase", units='-')
        figure.setRange(xRange = [-1.2,1.2])
        figure.setRange(yRange = [-1.2,1.2])
        
        return figure.plot(signal.real, signal.imag, pen=None, symbol='o', symbolPen=None, symbolSize=4, symbolBrush=(col[0]*255,col[1]*255,col[2]*255, 100))

   

def plot_filter_taps(figure,taps):
    figure.addLegend()

    figure.showGrid(x=True, y=True)
    figure.setLabel('left', "weight", units='-')
    figure.setLabel('bottom', "tap", units='-')
    finalBlock = taps.shape[0]-1
    print(taps.shape)
    return figure.plot(taps.real[finalBlock],pen = (0,255,0,255),name='real'), figure.plot(taps.imag[finalBlock],pen = (255,0,0,255),name='imag')

def plot_convergence(figure,error,phase,slipups,slipdowns):
   figure.addLegend()
   err_plt = figure.plot(error, pen=(255,0,0,255), name = 'error')
   phase_plt = figure.plot(phase,pen = (0,255,0,255), name = 'phase')
   slipups_plt = figure.plot(slipups,phase[slipups],pen=None, symbol='t', name = 'slipups')
   slipdowns_plt = figure.plot(slipdowns,phase[slipdowns],pen=None, symbol='t1', name = 'slipdowns')
   
   figure.showGrid(x=True, y=True)
   figure.setRange(yRange = [0,0.5])
   
    
class InteractiveMimoPlot :
    def __init__(self,win, request : BitMimoPlotRequest,lowerbound,upperbound):
        
        win.nextRow()
        self.convergence_figure = win.addPlot(title= request.name + " Convergence")
        plot_convergence(self.convergence_figure,request.error,request.phase,request.slipups,request.slipdowns)
        self.lr = pg.LinearRegionItem([lowerbound,upperbound])
        self.lr.setZValue(-10)
        self.convergence_figure.addItem(self.lr)
        self.N = len(request.error) 
        print(request.taps.shape[1])
        self.ntaps = request.taps.shape[1]


        self.constellation_figure = win.addPlot(title= request.name +" Constellation")
        self.list_constellation_plots = []
        for i_bit in range(len(request.bit_sigs)):
            constellation_plot = make_constelation_plot(self.constellation_figure,request.bit_sigs[i_bit],request.colors[i_bit])
            self.list_constellation_plots.append(constellation_plot)
     
        self.list_real_taps = []
        self.list_imag_taps = []
        for k in range(request.taps.shape[0]):
            fig = win.addPlot(title= request.name +" taps")
            real_taps, imag_taps = plot_filter_taps(fig,request.taps[k])
            self.list_real_taps.append(real_taps)
            self.list_imag_taps.append(imag_taps)
         
        print(self.list_real_taps)
        def updateRegion():
            self.lr.setRegion(constellation_figure.getViewBox().viewRange()[0])

        def updatePlot():
            
            r = self.lr.getRegion()
            r_min = int(r[0])
            r_max = int(r[1])
            len_err = len(request.error)
            for i_bit in range(len(self.list_constellation_plots)):
                len_arr = len(request.bit_sigs[i_bit])
                bitmin = int(r_min/len_err *len_arr )
                bitmax = int(r_max/ len_err * len_arr)
                self.list_constellation_plots[i_bit].setData(request.bit_sigs[i_bit].real[bitmin:bitmax],request.bit_sigs[i_bit].imag[bitmin:bitmax])               

            for i_input in range(len(self.list_imag_taps)):
                t_tap = r_min / self.N
                i_tap = min(max([0,int(t_tap * self.ntaps)]),request.taps.shape[1]-1)
                self.list_real_taps[i_input].setData(request.taps.real[i_input,i_tap])
                self.list_imag_taps[i_input].setData(request.taps.imag[i_input,i_tap])

           #updateRegion()

        self.lr.sigRegionChanged.connect(updatePlot)
        updatePlot()



def plot_interactive_mimo(convergence_plot_request_list : list,lowerbound,upperbound):


    app = QtGui.QApplication([])
    win = pg.GraphicsWindow(title="Mimo Plotter")
    print(len(convergence_plot_request_list))
    win.resize(600 + len(convergence_plot_request_list) * 300,300 * len(convergence_plot_request_list))
    win.setWindowTitle('Mimo Plotter')
    pg.setConfigOptions(antialias=True)
    
    for request in convergence_plot_request_list:
        convergencePlot = InteractiveMimoPlot(win,request,lowerbound,upperbound)
    QtGui.QApplication.instance().exec_()
        

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    N = 4 * (10 ** 4)
    x = np.arange(0,N,1)
    debugError = np.sin(x/N)
    
    real = np.random.rand(N)*2-1
    imag = np.random.rand(N)*2-1
    Constellation = real + 1j * imag
    request = [MimoPlotRequest(debugError,Constellation,"Test"), MimoPlotRequest(debugError,Constellation,"Test2")]
    plot_interactive_mimo(request,30000,31000)

    #if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        
