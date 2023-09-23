import dash_mantine_components as dmc
import numpy as np
from dash import Input, Output, callback, dcc, html
from dash_iconify import DashIconify

header = dmc.Header(
    height=70,
    px=25,
    children=[
        dmc.Stack(
            justify="center",
            style={"height": 70},
            children=dmc.Grid(
                children=[
                    dmc.Col(
                        span="content",
                        children=[
                            DashIconify(icon="mdi:frequency", width=40, height=40)
                        ],
                        pt=15,
                    ),
                    dmc.Col(
                        [
                            dmc.Title("Koopman Modes Dashboard", order=1),
                        ],
                        span="content",
                        pt=12,
                    ),
                ],
            ),
        )
    ],
)


def graph_component(name: str, id: str):
    return html.Div(
        [
            dmc.Title(name, order=3, align="center"),
            dmc.Center(
                style={"width": "100%"},
                children=[dcc.Graph(id=id, style={"align": "center"})],
            ),
        ]
    )


def frequency_range_selector(frequencies: list, id="freq_range_slider"):

    max_freq = np.max(frequencies)
    min_freq = np.min(frequencies)

    freq_range = max_freq - min_freq

    marks = [
        {"value": round(i, 2), "label": str(round(i, 2)) + " Hz"} for i in frequencies
    ]

    marks.append({"value": 0, "label": "0 Hz"})

    component = html.Div(
        [
            dmc.Title("Frequency range", align="center", order=4),
            dmc.RangeSlider(
                id=id,
                min=min_freq - freq_range * 0.1,
                max=max_freq + freq_range * 0.1,
                minRange=freq_range * 0.01,
                value=[min_freq, max_freq],
                marks=marks,
                step=freq_range * 0.01,
                precision=2,
                size="xl",
                color="blue",
                mt=10,
                mb=35,
            ),
        ]
    )
    return component


def plots_components(frequencies):
    plots = dmc.Container(
        size="lg",
        style={"marginTop": 20},
        children=[
            frequency_range_selector(frequencies),
            dmc.Grid(
                children=[
                    dmc.Col(
                        [graph_component("Eigenvalues", "eig-plot")],
                        span=6,
                    ),
                    dmc.Col(
                        [graph_component("Frequencies", "freq-plot")],
                        span=6,
                    ),
                    dmc.Col(
                        [graph_component("Koopman Modes", "modes-plot")],
                        span=12,
                    ),
                ]
            ),
        ],
    )
    return plots
