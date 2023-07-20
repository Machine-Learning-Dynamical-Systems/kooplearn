from kooplearn.visualizer.utils import create_frequency_plot, create_plot_eigs, create_2d_plot_modes, create_plot_modes
from kooplearn.models.base import BaseModel
import numpy as np
import pickle as pkl

class Visualizer:
    def __init__(self, koopman:BaseModel):
        self.operator = koopman

    def save(self, filepath):
        #self.operator.save(filepath)
        pass

    def load(self, filepath):
        #self.operator.load(filepath)
        pass

    def plot_freqs(self):
        if not hasattr(self, 'eigs'):
            self.eigs = self.operator.eig()
        fig = create_frequency_plot(self.eigs)
        return fig

    def plot_eigs(self):
        if not hasattr(self, 'eigs'):
            self.eigs = self.operator.eig()
        fig = create_plot_eigs(self.eigs)
        return fig

    def plot_modes(self, index=None, image=False, coordinates=None):
        # plot moduluses and arguments
        # 
        modes = self.operator.modes()
        if image:
            fig = create_2d_plot_modes(modes, coordinates, index)
        else:
            fig = create_plot_modes(modes, index)
        return fig
