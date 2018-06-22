from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
from pyqt_mimo.mimofigure import *
from pyqt_mimo.convergenceplot import *
from pyqt_mimo.constellationplot import *
from pyqt_mimo.tapsplot import *   
    
class InteractiveMimoRow :
    def __init__(self,win, mimofigures,lowerbound,upperbound):
        win.nextRow()
        for fig in mimofigures:
            fig.create_figure(win)

        self.lr = pg.LinearRegionItem([lowerbound,upperbound])
        self.lr.setZValue(-10)
        mimofigures[0].fig.addItem(self.lr)
        self.mimofigures = mimofigures
        def updateRegion():
            self.lr.setRegion(constellation_figure.getViewBox().viewRange()[0])
        def updatePlot():
            r = self.lr.getRegion()
            r_min = int(r[0])
            r_max = int(r[1])
            for fig in mimofigures:
                fig.update_region( r_min, r_max)
        self.lr.sigRegionChanged.connect(updatePlot)
        updatePlot()



def plot_interactive_mimo(mimo_figures_rows : list,lowerbound,upperbound):

    app = QtGui.QApplication([])
    win = pg.GraphicsWindow(title="Mimo Plotter")
    win.resize(600 + len(mimo_figures_rows) * 300,300 * len(mimo_figures_rows))
    win.setWindowTitle('Mimo Plotter')

    pg.setConfigOptions(antialias=True)
    
    for figure_row in mimo_figures_rows:
        row = InteractiveMimoRow(win,figure_row,lowerbound,upperbound)
    QtGui.QApplication.instance().exec_()
        

