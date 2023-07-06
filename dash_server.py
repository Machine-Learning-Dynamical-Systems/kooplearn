from dash import Dash, dcc, html, Input, Output, callback
from kooplearn.visualizer.visualizer import Visualizer
import numpy as np
import argparse

#### WORK IN PROGRESS, FOR NOW USE THE visualizer.utils.py METHODS ####
# Script for an interactive Dash html page using the visualisation methods of utils.py

parser=argparse.ArgumentParser(
    description="Dash web application for visualisation of a Koopman operator"
)
parser.add_argument( # location of the koopman estimator
    '--koopman',
    type=str
)

args = parser.parse_args()
operator = np.load(args.koopman)

viz = Visualizer(operator)

app = Dash(__name__)

app.layout(
    html.Div([
        html.H4('Koopman Visualisation'),
        html.P("Operator's eigenvalues:"),
        dcc.Graph(id="eig-plot"),
        html.P("Frequency plot:"),
        dcc.Graph(id='freq-plot')])
    )

@callback(
    Output('eig-plot', 'figure'),
    Input()
)
def update_eig_plot():
    return viz.create_plot_eigs()

@callback(
    Output('freq-plot', 'figure'),
    Input()
)
def update_freq_plot():
    return viz.create_frequency_plot()

app.run_server(debug=True)