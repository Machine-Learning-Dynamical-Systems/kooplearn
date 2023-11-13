import argparse
import pickle

import numpy as np
from dash import Dash, Input, Output, callback, dcc, html
from sklearn.gaussian_process.kernels import DotProduct

from kooplearn.dashboard.visualizer import Visualizer
from kooplearn.datasets import Mock
from kooplearn.models.kernel import KernelDMD
from kooplearn.data import traj_to_contexts

#### WORK IN PROGRESS, FOR NOW USE THE visualizer.utils.py METHODS ####
# Script for an interactive Dash html page using the visualisation methods of utils.py

parser = argparse.ArgumentParser(
    description="Dash web application for visualisation of a Koopman operator"
)
parser.add_argument(  # location of the koopman estimator
    "--koopman", type=str, default=""
)

args = parser.parse_args()
if args.koopman == "":
    # tutorial mode
    dataset = Mock(num_features=10, rng_seed=0)
    _Z = dataset.sample(None, 100)
    X = traj_to_contexts(_Z, 2)

    operator = KernelDMD(DotProduct(), rank=10)
    operator.fit(X)
else:
    with open(args.koopman, "rb") as file:
        operator = pickle.load(file)

viz = Visualizer(operator)
available_modes = viz.infos["eig_num"].unique().astype("str")
available_modes = np.insert(available_modes, 0, "All")
available_modes = np.insert(available_modes, 1, "Combined")

frequencies = viz.infos["frequency"].unique()
pos_frequencies = frequencies[frequencies > 0]
frequency_dict = {i: str(round(i, 3)) for i in pos_frequencies}
frequency_dict[0] = "0"

app = Dash(__name__)

app.layout = html.Div(
    [
        html.Div(
            [dcc.Graph(id="eig-plot")],
            style={"width": "30%", "display": "inline-block"},
        ),
        html.Div(
            [dcc.Graph(id="freq-plot")],
            style={"width": "49%", "display": "inline-block"},
        ),
        html.Div(
            [
                dcc.RangeSlider(
                    min=-0.1,
                    max=int(viz.infos["frequency"].max()) + 1,
                    marks=frequency_dict,
                    id="freq_range_slider",
                ),
                dcc.Input(id="Tmax", type="number", placeholder="input T max"),
                dcc.Slider(min=0, max=1, step=1, id="T"),
            ],
            style={"width": "60%"},
        ),
        html.Div(
            [
                html.H4("Modes"),
                dcc.Dropdown(available_modes, "All", id="modes_select"),
                dcc.Graph(id="modes-plot"),
                # html.H1("Prediction"),
                # dcc.Graph(id='pred-plot', figure=viz.plot_preds())
            ]
        ),
    ]
)


@callback(
    Output("eig-plot", "figure"),
    Output("freq-plot", "figure"),
    Output("modes-plot", "figure"),
    Output("T", "max"),
    Input("freq_range_slider", "value"),
    Input("Tmax", "value"),
    Input("T", "value"),
    Input("modes_select", "value"),
)
def update_modes_plots(value, Tmax, T, mode_selection):
    if value is None:
        min_freq = viz.infos["frequency"].unique().min()
        max_freq = viz.infos["frequency"].unique().max()
    else:
        min_freq = value[0]
        max_freq = value[1]

    if T is None:
        T = 1

    fig_eig = viz.plot_eigs(min_freq, max_freq)
    fig_freqs = viz.plot_freqs(min_freq, max_freq)

    if mode_selection == "All":
        fig_modes = viz.plot_modes(index=None, min_freq=min_freq, max_freq=max_freq)
    elif mode_selection == "Combined":
        fig_modes = viz.plot_combined_modes(T, min_freq, max_freq)
    else:
        fig_modes = viz.plot_modes(
            index=[int(mode_selection)], min_freq=min_freq, max_freq=max_freq
        )
    # fig_pred = viz.plot_preds(operator.X_fit_[-1], 1, min_freq, max_freq)
    return fig_eig, fig_freqs, fig_modes, Tmax


# TODO:
#  Predictions at t+n, where we can select n and filter by frequency
#  Modes filtered by frequency

# https://dash.plotly.com/dash-core-components/slider
# https://plotly.com/python/filter/
# https://stackoverflow.com/questions/45736656/how-to-use-a-button-to-trigger-callback-updates

app.run_server(debug=True)
