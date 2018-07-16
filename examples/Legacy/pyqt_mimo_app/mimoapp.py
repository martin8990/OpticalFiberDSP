from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import *

class Errorcalc_picker(QWidget):
    def __init__(self, parent = None):
      super().__init__(parent)		
      layout = QHBoxLayout()
      layout.addWidget(QLabel( "Error calculation")  )
      self.b1 = QRadioButton("CMA")
      self.b1.setChecked(True)
      layout.addWidget(self.b1)		
      self.b2 = QRadioButton("LMS")
      layout.addWidget(self.b2)
      self.setLayout(layout)

class numeric_field(QWidget):
    def __init__(self,name,initial_value, parent = None):
      super().__init__(parent)		
      layout = QHBoxLayout()
      layout.addWidget(QLabel(name)  )
      self.b1 = QTextEdit(str(initial_value))
      self.b1.resize(100,20)
      layout.addWidget(self.b1)		
      
      self.setLayout(layout)
      

class MimoControllerApp():
    def __init__(self): 
        app = QtGui.QApplication([])
        s = QStyleFactory.create('Fusion')
        app.setStyle(s)
        # set new style globally
        w = QWidget()
        w.resize(600, 600)
        l = QFormLayout()

        self.mu = QTextEdit('1e-3')
        self.lb = QTextEdit('64')
        self.error_calc_picker = Errorcalc_picker()

        l.addRow(QLabel('Stepsize (mu)'),self.mu)
        l.addRow(QLabel('Mimo block length (lb)'),self.lb)
        l.addChildWidget(self.error_calc_picker)

        l.addWidget(QPushButton('Rerun mimo'))
        #l.addWidget(QCheckBox('Oversampled'))
    

        #l.addWidget(numeric_field("Mimo Block Length",64))
        #l.addWidget(numeric_field("mu",1e-3))


        #l.addWidget(Errorcalc_picker())
        w.setLayout(l)
        w.show()
        QtGui.QApplication.instance().exec_()
    
if __name__ == "__main__":
    MimoControllerApp()
