from pyqt_mimo.mimoplot import MimoFigure
import numpy as np
class ConvergencePlot(MimoFigure):
    def __init__(self,error,name,ntrainingsyms = None,nloops = 0, phase = np.ones(1)*999,slipups = np.ones(1)*999,slipdowns = np.ones(1)*999):
        self.error = error
        self.name = name
        self.ntrainingsyms = ntrainingsyms
        self.nloops= nloops
        self.phase = phase
        self.slipups = slipups
        self.slipdowns = slipdowns

    def create_figure(self, win):
       self.fig = win.addPlot(title= self.name + " Convergence")
       self.fig.addLegend()

       err_plt = self.fig.plot(self.error.real, pen=(255,0,0,255), name = 'real error')
       #err_plt = self.fig.plot(self.error.imag, pen=(255,0,0,255), name = 'imag error')
       
       if self.ntrainingsyms!=None:


           ids = (np.arange(self.nloops + 1)+1) * self.ntrainingsyms
           for k in range(ids.shape[0]):
               ids[k] = min(ids[k],len(self.error)-1)

           train_plt=  self.fig.plot(ids,self.error[ids],pen=None, symbol='star', name = 'training_stop')
      
       if self.phase[0]!=999:
           phase_plt = self.fig.plot(self.phase,pen = (0,255,0,255), name = 'phase')
           if self.slipdowns[0]!=999:
               slipups_plt = self.fig.plot(self.slipups,self.phase[self.slipups],pen=None, symbol='t', name = 'slipups')
               slipdowns_plt = self.fig.plot(self.slipdowns,self.phase[self.slipdowns],pen=None, symbol='t1', name = 'slipdowns')
   
       self.fig.showGrid(x=True, y=True)
       self.fig.setRange(yRange = [0,0.5])
    def update_region(self, r_min, r_max):
        pass
