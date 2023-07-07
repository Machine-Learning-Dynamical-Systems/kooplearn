from dash import Dash, dcc, html, Input, Output, callback
from utils import create_frequency_plot, create_plot_eigs
from kooplearn.models.base import BaseModel


# Script for an interactive Dash html page using the visualisation methods of utils.py

class Visualizer:
    def save(self, koopman:BaseModel):
        self.operator = koopman

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
