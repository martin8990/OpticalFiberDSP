from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import *

class ConvergencePlotWidget(QWidget):
    def __init__(self,error,phase,slipups,slipdowns, parent = None):
      super().__init__(parent)		
      self.error = error
      self.phase = phase
      self.slipups = slipups
      self.slipdowns = slipdowns
      self.modetoggles = []

      layout = QHBoxLayout()
      layout.addWidget(QLabel("ConvergencePlotWidget")  )
      for i_mode in range(len(error)):
          self.modetoggles.append(QCheckBox("Mode " +str(i_mode)))
          if i_mode<3:
              self.modetoggles[i_mode].setChecked(True)
          layout.addWidget(self.modetoggles[i_mode])		
      self.setLayout(layout)

class TapsPlotWidget(QWidget):
    def __init__(self,taps, parent = None):
      super().__init__(parent)		
      self.taps = taps
      self.modetoggles = []

      layout = QGridLayout()

      for i_input in range(taps.shape[0]):
          self.modetoggles.append([])
          for i_output in range(taps.shape[1]):
              self.modetoggles[i_input].append(QCheckBox("Tap " +str(i_input)+ " to " + str(i_output)))
              layout.addWidget(self.modetoggles[i_input][i_output],i_input,i_output)		
          self.modetoggles[i_input][i_output].setChecked(True)
             
      self.setLayout(layout)



class PlotControllerApp():
    def __init__(self): 
        self.app = QtGui.QApplication([])
        s = QStyleFactory.create('Fusion')
        self.app.setStyle(s)
        # set new style globally
        self.w = QWidget()
        self.w.resize(600, 600)
        
        self.l = QFormLayout()
        self.l.addWidget(QPushButton('Rebuild Plots'))
    def add_widget(self,widget):
        self.l.addWidget(widget)
    def complete(self):
        self.w.setLayout(self.l)
        self.w.show()
        QtGui.QApplication.instance().exec_()


#import pyqtgraph.examples
#pyqtgraph.examples.run()