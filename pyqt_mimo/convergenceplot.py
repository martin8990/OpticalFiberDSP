from pyqt_mimo.mimoplot import MimoFigure
class ConvergencePlot(MimoFigure):
    def __init__(self,error,phase,slipups,slipdowns,name):
        self.error = error
        self.phase= phase
        self.slipups = slipups
        self.slipdowns = slipdowns
        self.name = name

    def create_figure(self, win):
       self.fig = win.addPlot(title= self.name + " Convergence")
       self.fig.addLegend()
       err_plt = self.fig.plot(self.error, pen=(255,0,0,255), name = 'error')
       phase_plt = self.fig.plot(self.phase,pen = (0,255,0,255), name = 'phase')
       slipups_plt = self.fig.plot(self.slipups,self.phase[self.slipups],pen=None, symbol='t', name = 'slipups')
       slipdowns_plt = self.fig.plot(self.slipdowns,self.phase[self.slipdowns],pen=None, symbol='t1', name = 'slipdowns')
   
       self.fig.showGrid(x=True, y=True)
       self.fig.setRange(yRange = [0,0.5])
    def update_region(self, r_min, r_max):
        pass

class ConvergenceErrorPhasePlot(MimoFigure):
    def __init__(self,error,phase,name):
        self.error = error
        self.phase= phase
        self.name = name

    def create_figure(self, win):
       self.fig = win.addPlot(title= self.name + " Convergence")
       self.fig.addLegend()
       err_plt = self.fig.plot(self.error, pen=(255,0,0,255), name = 'error')
       phase_plt = self.fig.plot(self.phase,pen = (0,255,0,255), name = 'phase')
       
       self.fig.showGrid(x=True, y=True)
       self.fig.setRange(yRange = [0,0.5])
    def update_region(self, r_min, r_max):
        pass


class ConvergencePlotBasic(MimoFigure):
    def __init__(self,error,name):
        self.error = error
        self.name = name

    def create_figure(self, win):
       self.fig = win.addPlot(title= self.name + " Convergence")
       self.fig.addLegend()
       err_plt = self.fig.plot(self.error, pen=(255,0,0,255), name = 'error')
    
       self.fig.showGrid(x=True, y=True)
       self.fig.setRange(yRange = [0,0.5])
    def update_region(self, r_min, r_max):
        pass
