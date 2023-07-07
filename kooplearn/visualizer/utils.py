import plotly.express as px
import numpy as np


def create_plot_eigs(eigs):
    real = eigs.real
    comp = eigs.imag

    fig = px.scatter(x=real, y=comp)
    fig.add_shape(type='circle',
                  fillcolor='PaleTurquoise',
                  line_color='LightSeaGreen',
                  opacity=0.3,
                  x0=-1, y0=-1, x1=1, y1=1)

    return fig


def create_frequency_plot(eigs, deltat=1):
    # t is \Delta t in Kutz et al.
    modulus = np.abs(eigs)
    angles = np.log(eigs)
    freqs = angles.imag / (2 * np.pi * deltat)

    fig = px.scatter(x=freqs, y=modulus)
    fig.add_bar(x=freqs, y=modulus, width=0.001)

    return fig


def create_plot_modes():
    pass
