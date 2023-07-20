import plotly.express as px
from plotly.subplots import make_subplots
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
    fig.update_layout(
        xaxis_title="Real", yaxis_title="Imaginary",
        width=800, height=800,
        title='Eigenvalues'
    )
    return fig

def create_frequency_plot(eigs, deltat=1):
    # t is \Delta t in Kutz et al.
    modulus = np.abs(eigs)**(1/deltat)
    angles = np.log(eigs)
    freqs = angles.imag / (2 * np.pi * deltat)
    index = (freqs >=0)

    fig = px.scatter(x=freqs[index], y=modulus[index])
    fig.add_bar(x=freqs[index], y=modulus[index], width=0.001)
    fig.update_layout(
        xaxis_title="Frequency", yaxis_title="Amplitude",
        title='Frequency plot'
    )
    # version with amplitude multiplied with mode modulus

    return fig

# TODO: function comparing koopman to fourier

def create_plot_modes(modes, index=None):
    # image is false if data one dimensional, true if two dimensional
    # coordinates is hte position of each pixel if two dimensional
    # index is None if all modes are to be plotted, an array if some nodes must plotted or an int if one mode must be plotted
    if index is None:
        index = range(modes.shape[0])
    n_plots = len(index)
    if n_plots%2 == 0:
        fig = make_subplots(rows = n_plots//2, cols=2, x_title='Variable', y_title='Signal')
    else:
        fig = make_subplots(rows = n_plots//2+1, cols=2)
    for i in index:
        subfig = px.scatter(x = range(modes.shape[1]), y=modes[i].real, title='mode number {}'.format(i+1))
        fig.add_trace(subfig.data[0], col=i%2+1, row=i//2+1)
    fig.update_layout(title='Mode decomposition')
    return fig

def create_2d_plot_modes(modes, coordinates, index=None):
    if index is None:
        index = range(modes.shape[0])
    n_plots = len(index)
    if n_plots%2 == 0:
        fig = make_subplots(rows = n_plots//2, cols=2)
    else:
        fig = make_subplots(rows = n_plots//2+1, cols=2)
    for i in index:
        fig.add_trace(px.scatter(x=coordinates[:,0], y=coordinates[:,1], color=modes[i]), col=i%2, row=i//2)
    fig.update_layout(xaxis_title='Variable', yaxis_title='Signal')
    return fig

def map_predictions(preds, coordinates, height=False):
    # coordinates is a list of x,y coordinates in np array format
    if not height:
        fig = px.scatter(x=coordinates[:,0], y=coordinates[:,1], color=preds)
    else:
        fig = px.scatter_3d(x=coordinates[:,0], y=coordinates[:,1], z=preds, color=preds)
    return fig

def plot_predictions(preds, names):
    fig = px.line(y=preds, x=names)
    fig.update_layout(
        xaxis_title='Variables', yaxis_title='Predicted value'
    )
    return fig

# TODO: plot x= time, y= the mode frequencies, z = amplitudes