# -*- coding: utf-8 -*-
"""
This example demonstrates the use of GLSurfacePlotItem.
"""

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np

def create_heatmap(dim,ZZ):

    ## Create a GL View widget to display data
    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.show()
    w.setWindowTitle('pyqtgraph example: GLSurfacePlot')
    w.setCameraPosition(distance=50)

    ## Add a grid to the view
    g = gl.GLGridItem()
    g.scale(2,2,1)
    g.setDepthValue(10)  # draw grid after surfaces since they may be translucent
    w.addItem(g)

    ## Saddle example with x and y specified



    p2 = gl.GLSurfacePlotItem(x=dim*10, y=dim*10, z=ZZ/10, shader='heightColor')
    w.addItem(p2)
    QtGui.QApplication.instance().exec_()





## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
