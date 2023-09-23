from functools import partial
from typing import Optional

from dash import Input, Output, callback

from kooplearn._src.dashboard.visualizer import Visualizer


def update_modes_visibility(viz: Visualizer, frequency_range: Optional[list] = None):
    if frequency_range is None:
        min_freq = viz.infos["frequency"].unique().min()
        max_freq = viz.infos["frequency"].unique().max()
    else:
        min_freq = frequency_range[0]
        max_freq = frequency_range[1]

    fig_eig = viz.plot_eigs(min_freq, max_freq)
    fig_freqs = viz.plot_freqs(min_freq, max_freq)
    fig_modes = viz.plot_modes(index=None, min_freq=min_freq, max_freq=max_freq)
    return fig_eig, fig_freqs, fig_modes


def get_cb_functions(
    viz: Visualizer,
    eig_id="eig-plot",
    freq_id="freq-plot",
    modes_id="modes-plot",
    frequency_range_id="freq_range_slider",
):
    update_modes_visibility_partial = partial(update_modes_visibility, viz)
    return callback(
        Output(eig_id, "figure"),
        Output(freq_id, "figure"),
        Output(modes_id, "figure"),
        Input(frequency_range_id, "value"),
    )(update_modes_visibility_partial)
