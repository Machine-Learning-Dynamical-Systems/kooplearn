import argparse
import pickle
import time

import numpy as np
from dash import Dash, Input, Output, callback, dcc, html
from dash.dependencies import ALL, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px

from sklearn.gaussian_process.kernels import *
from kooplearn.models.feature_maps import *
from featuremap_example import feature_map

from kooplearn.dashboard.visualizer import Visualizer
from kooplearn.datasets import *
from kooplearn.models import *
from kooplearn.data import traj_to_contexts

from scipy.stats import ortho_group
import numpy as np

# stylesheet with the .dbc class to style  dcc, DataTable and AG Grid components with a Bootstrap theme
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME, dbc_css])

header = html.H4(
    "Dash web application for visualisation of a Koopman operator", className="bg-primary text-white p-2 mb-2 text-center"
)

progress_bar = dbc.Progress(id="progress-bar", striped=True, animated=True, style={"height": "20px", "visibility": "hidden"}, value=100)
# spinner = dbc.Spinner(size="lg", color="primary", id="loading-spinner")

update_button = html.Div([dbc.Button("Update Plots", id="update-button", className="mb-3", color="primary", n_clicks=1), progress_bar])

# Define your controls using Dash Bootstrap Components
models = dbc.Tab([html.Div(
            [
                # html.H4("Models"),
                dbc.Label("Select a model"),
                dcc.Dropdown(
                    id="model-dropdown",
                    options=[
                        {"label": "KernelDMD", "value": "KernelDMD"},
                        {"label": "DeepEDMD", "value": "DeepEDMD"},
                        {"label": "DMD", "value": "DMD"},
                        {"label": "ExtendedDMD", "value": "ExtendedDMD"},                        
                    ],
                    value="KernelDMD",  # Default model selection
                    clearable=False
                ),],
                className="p-3",),
                html.Div([
                dbc.Label("Rank: ", style={"display": "inline-block", "margin-right": "8px"}),
                dcc.Input(id="rank-input", type="number", min=1, step=1, placeholder="rank", value=1),], className="p-3"),  # Assuming rank starts at 1 and increments by 1
                
                html.Div([
                dbc.Label("Tikhonov Regularization: ", style={"display": "inline-block", "margin-right": "8px"}),
                dcc.Input(id="tikhonov_reg-input", type="number", placeholder="tikhonov_reg"),], className="p-3",),

                html.Div([
                dbc.Label("context_window_len: ", style={"display": "inline-block", "margin-right": "8px"}),
                dcc.Input(id="context_window_len-input", type="number", min=2, step=1, value=2, placeholder="context_window_len"),], className="p-3",),

                html.Div(id='model-params-div'),
        ],
        label="Models"
        )

datasets = dbc.Tab([html.Div(
            [
                # html.H4("Datasets"),
                dbc.Label("Select a dataset"),
                dcc.Dropdown(
                    id="dataset-dropdown",
                    options=[
                        # {"label": "Mock", "value": "Mock"},
                        {"label": "LinearModel", "value": "LinearModel"},
                        {"label": "LogisticMap", "value": "LogisticMap"},
                        {"label": "LangevinTripleWell1D", "value": "LangevinTripleWell1D"},
                        {"label": "DuffingOscillator", "value": "DuffingOscillator"},
                        {"label": "Lorenz63", "value": "Lorenz63"},
                    ],
                    value="LinearModel",
                    clearable=False
                ),
                html.Div(id='dataset-params-div'),  # This Div will be populated with inputs dynamically
        
                dbc.Label("T parameter of the sample function: ", style={"display": "inline-block", "margin-right": "8px"}),
                dcc.Input(id="T_sample-input", type="number", min=1, step=1, value=100, placeholder="T"),
            ],
            className="p-3",
        )],
        label="Datasets"
        )

slider1 = html.Div(
            [
                dbc.Label("Frequency"),
                dcc.RangeSlider(
                    min=-0.1,
                    max= 1, #int(viz.infos["frequency"].max()) + 1,
                    marks= {"0": "0", "1": "1"}, #frequency_dict,
                    id="freq_range_slider",
                    tooltip={"placement": "bottom", "always_visible": True},
                    className="p-0",
                ),
            ],
            className="p-3",
)

slider2 = html.Div(
            [
                # dbc.Label("T"),
                dcc.Input(id="Tmax", type="number", placeholder="input T max"),
                dcc.Slider(min=0, 
                           max=1, 
                           step=1, 
                           id="T",
                           tooltip={"placement": "bottom", "always_visible": True},
                           className="p-0",
                    ),
            ],
            className="p-3",
)

modes = dbc.Tab([html.Div(
            [
                # dcc.Dropdown(available_modes, "All", id="modes_select"),
                html.H4("Modes"),
                dbc.Label("Select a mode"),
                dcc.Dropdown(
                    id="modes_select",
                    options=[{"label": str(i), "value": str(i)} for i in range(10)],  # Placeholder values
                    value="All",  # Placeholder value
                    clearable=False
                ),
                dcc.Graph(id="modes-plot"),
                # html.H1("Prediction"),
                # dcc.Graph(id='pred-plot', figure=viz.plot_preds())
            ],
            className="p-3",
        )], label="Modes")

controls = dbc.Tab([html.Div(
    [slider1, slider2],
    className="p-3",
)],label="Frequency and T")


tabs_control = dbc.Card(dbc.Tabs([controls, models, datasets]), body=True,)

graph1 = dbc.Col([html.Div(
            [html.H4("Eigenvalues plot"),
                dcc.Graph(id="eig-plot"),
                ],
                )], align="center", width=5)                
            # style={"width": "30%", "display": "inline-block"},
            # style={"padding": "20px", "boxShadow": "0 0 10px #ccc"},

graph2 = dbc.Col([html.Div(
            [
                html.H4("Frequency plot"),
                dcc.Graph(id="freq-plot"),
                ],
                # style={"width": "49%", "display": "inline-block"},
                )], align="center", width=7)

dataset_plot = dbc.Tab([html.Div(
            [
                dcc.Graph(id="dataset-plot"),
            ],
            className="p-3",
        )], label="Dataset Plot")

tab_plots = dbc.Tab(dbc.Row([graph1, graph2]), label="Plots")

tabs_graphs = dbc.Card(dbc.Tabs([dataset_plot, tab_plots, modes]), body=True)

# graphs = dbc.Card([graphs1])

app.layout = dbc.Container(
    [
        header,
        # html.Div(html.Img(src="https://kooplearn.readthedocs.io/en/latest/_static/logo.svg", height="120px"), 
        #         #  className="d-flex justify-content-center"
        #          ),
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col([html.Img(src="https://kooplearn.readthedocs.io/en/latest/_static/logo.svg", height="120px")], align="center"),
                    dbc.Col([update_button], align="center"),
                    ]),
                # controls,
                tabs_control
            ],  width=4),
            dbc.Col([tabs_graphs], width=8),
            # dbc.Col([modes], width=7)
            # dbc.Col([graph2], width=5),
        ]),
        html.Div(id="click-count", style={"display": "none"}, children="1"),
    ], 
    fluid=True,
    className="bg-light",
)


# Dictionary of dataset parameters
dataset_params = {
    'LinearModel': [
        {'label': 'Noise', 'id': {'type': 'dynamic-param', 'index': 'noise'}, 'value': 1., 'step': 0.01},
        {'label': 'RNG Seed', 'id': {'type': 'dynamic-param', 'index': 'rng_seed'}, 'value': None, 'step': 1},
        {'label': 'Number features', 'id': {'type': 'dynamic-param', 'index': 'num_features'}, 'value': 10, 'step': 1},
        {'label': 'Rang de la matrice', 'id': {'type': 'dynamic-param', 'index': 'r'}, 'value': 5, 'step': 1},
        {'label': 'Lambda', 'id': {'type': 'dynamic-param', 'index': 'l'}, 'value': 1., 'step': 0.1}
    
    ],
    'LogisticMap': [
        {'label': 'r', 'id': {'type': 'dynamic-param', 'index': 'r'}, 'value': 4.0, 'step': 0.1},
        {'label': 'N', 'id': {'type': 'dynamic-param', 'index': 'N'}, 'value': None, 'step': 1},
        {'label': 'RNG Seed', 'id': {'type': 'dynamic-param', 'index': 'rng_seed'}, 'value': None, 'step': 1}
    ],
    'LangevinTripleWell1D': [
        {'label': 'gamma', 'id': {'type': 'dynamic-param', 'index': 'gamma'}, 'value': 0.1, 'step': 0.01},
        {'label': 'kt', 'id': {'type': 'dynamic-param', 'index': 'kt'}, 'value': 1.0, 'step': 0.1},
        {'label': 'dt', 'id': {'type': 'dynamic-param', 'index': 'dt'}, 'value': 1e-4, 'step': 1e-5},
        {'label': 'RNG Seed', 'id': {'type': 'dynamic-param', 'index': 'rng_seed'}, 'value': None, 'step': 1}
    ],
    'DuffingOscillator': [
        {'label': 'alpha', 'id': {'type': 'dynamic-param', 'index': 'alpha'}, 'value': 0.5, 'step': 0.01},
        {'label': 'beta', 'id': {'type': 'dynamic-param', 'index': 'beta'}, 'value': 0.0625, 'step': 0.0001},
        {'label': 'gamma', 'id': {'type': 'dynamic-param', 'index': 'gamma'}, 'value': 0.1, 'step': 0.1},
        {'label': 'delta', 'id': {'type': 'dynamic-param', 'index': 'delta'}, 'value': 2.5, 'step': 0.1},
        {'label': 'omega', 'id': {'type': 'dynamic-param', 'index': 'omega'}, 'value': 2.0, 'step': 0.1},
        {'label': 'dt', 'id': {'type': 'dynamic-param', 'index': 'dt'}, 'value': 0.01, 'step': 0.01}
    ],
    'Lorenz63': [
        {'label': 'sigma', 'id': {'type': 'dynamic-param', 'index': 'sigma'}, 'value': 10, 'step': 1},
        {'label': 'mu', 'id': {'type': 'dynamic-param', 'index': 'mu'}, 'value': 28, 'step': 1},
        {'label': 'beta', 'id': {'type': 'dynamic-param', 'index': 'beta'}, 'value': 8 / 3, 'step': 0.01},
        {'label': 'dt', 'id': {'type': 'dynamic-param', 'index': 'dt'}, 'value': 0.01, 'step': 0.01}
    ]
}

def create_dataset_params(dataset_name):
    # Get parameters for the selected dataset
    params = dataset_params.get(dataset_name, [])
    
    # Create input components for each parameter
    children = []
    for param in params:
        children.append(html.Div([
            dbc.Label(f"{param['label']}: ", style={"display": "inline-block", "margin-right": "8px"},),
            dcc.Input(id=param['id'], type="number", value=param['value'], step=param['step'], placeholder=param['label']),
        ], className="p-2"))

    return children



model_params = {
        'KernelDMD': [
            {'label': 'Reduced Rank', 'id': {'type': 'model-param', 'index': 'reduced_rank'}, 'type': 'checklist', 'default': True},
            # {'label': 'Rank', 'id': {'type': 'model-param', 'index': 'rank'}, 'type': 'number', 'default': 5},
            # {'label': 'Tikhonov Regularization', 'id': {'type': 'model-param', 'index': 'tikhonov_reg'}, 'type': 'number', 'default': None},
            {'label': 'SVD Solver', 'id': {'type': 'model-param', 'index': 'svd_solver'}, 'type': 'dropdown', 'options': ['full', 'arnoldi', 'randomized'], 'default': 'full'},
            {'label': 'Iterated Power', 'id': {'type': 'model-param', 'index': 'iterated_power'}, 'type': 'number', 'default': 1},
            # {'label': 'N Oversamples', 'id': {'type': 'model-param', 'index': 'n_oversamples'}, 'type': 'number', 'default': 5},
            {'label': 'Kernel', 'id': {'type': 'model-param', 'index': 'kernel'}, 'type': 'dropdown', 'options': ['DotProduct', 'RBF', 'Matern', '0.5*DotProduct + 0.5*RBF'], 'default': 'DotProduct'}, #'options': ['DotProduct', 'Exponentiation', 'PairwiseKernel', 'Sum', 'Product']
            {'label': 'Length Scale', 'id': {'type': 'model-param', 'index': 'length_scale'}, 'type': 'number', 'default': 1.0},
            # {'label': 'Optimal Sketching', 'id': {'type': 'model-param', 'index': 'optimal_sketching'}, 'type': 'checklist', 'default': False},
            {'label': 'RNG Seed', 'id': {'type': 'model-param', 'index': 'rng_seed'}, 'type': 'number', 'default': None}
        ],
        'DeepEDMD': [
            {'label': 'Maximum number of epochs', 'id': {'type': 'model-param', 'index': 'max_epochs'}, 'type': 'number', 'default': 10},
            {'label': 'Reduced Rank', 'id': {'type': 'model-param', 'index': 'reduced_rank'}, 'type': 'checklist', 'default': True},
            {'label': 'SVD Solver', 'id': {'type': 'model-param', 'index': 'svd_solver'}, 'type': 'dropdown', 'options': ['full', 'arnoldi', 'randomized'], 'default': 'full'},
            {'label': 'Iterated Power', 'id': {'type': 'model-param', 'index': 'iterated_power'}, 'type': 'number', 'default': 1},
            # {'label': 'N Oversamples', 'id': {'type': 'model-param', 'index': 'n_oversamples'}, 'type': 'number', 'default': 5},
            {'label': 'RNG Seed', 'id': {'type': 'model-param', 'index': 'rng_seed'}, 'type': 'number', 'default': None}
        ],
        'ExtendedDMD': [
            {'label': 'Reduced Rank', 'id': {'type': 'model-param', 'index': 'reduced_rank'}, 'type': 'checklist', 'default': True},
            {'label': 'SVD Solver', 'id': {'type': 'model-param', 'index': 'svd_solver'}, 'type': 'dropdown', 'options': ['full', 'arnoldi', 'randomized'], 'default': 'full'},
            {'label': 'Iterated Power', 'id': {'type': 'model-param', 'index': 'iterated_power'}, 'type': 'number', 'default': 1},
            # {'label': 'N Oversamples', 'id': {'type': 'model-param', 'index': 'n_oversamples'}, 'type': 'number', 'default': 5},
            {'label': 'FeatureMap', 'id': {'type': 'model-param', 'index': 'feature_map'}, 'type': 'dropdown', 'options': ['IdentityFeatureMap'], 'default': 'IdentityFeatureMap'},
            {'label': 'RNG Seed', 'id': {'type': 'model-param', 'index': 'rng_seed'}, 'type': 'number', 'default': None}
        ],
        'DMD': [
            {'label': 'Reduced Rank', 'id': {'type': 'model-param', 'index': 'reduced_rank'}, 'type': 'checklist', 'default': True},
            {'label': 'SVD Solver', 'id': {'type': 'model-param', 'index': 'svd_solver'}, 'type': 'dropdown', 'options': ['full', 'arnoldi', 'randomized'], 'default': 'full'},
            {'label': 'Iterated Power', 'id': {'type': 'model-param', 'index': 'iterated_power'}, 'type': 'number', 'default': 1},
            # {'label': 'N Oversamples', 'id': {'type': 'model-param', 'index': 'n_oversamples'}, 'type': 'number', 'default': 5},
            {'label': 'RNG Seed', 'id': {'type': 'model-param', 'index': 'rng_seed'}, 'type': 'number', 'default': None}
        ],
    }

def create_model_params(model_name):
    params = model_params.get(model_name, [])

    children = []
    for param in params:
        if param['type'] == 'number':
            children.append(html.Div([
                dbc.Label(f"{param['label']}: ", style={"display": "inline-block", "margin-right": "8px"}),
                dcc.Input(id=param['id'], type="number", value=param['default'], placeholder=param['label']),
            ], className="p-2"))
        elif param['type'] == 'dropdown':
            children.append(html.Div([
                dbc.Label(f"{param['label']}: ", style={"display": "inline-block", "margin-right": "8px"}),
                dcc.Dropdown(id=param['id'], options=[{'label': val, 'value': val} for val in param['options']], value=param['default'], clearable=False),
            ], className="p-2"))
        elif param['type'] == 'checklist':
            children.append(html.Div([
                dbc.Label(f"{param['label']}: ", style={"display": "inline-block", "margin-right": "8px"}),
                dcc.RadioItems(id=param['id'], options=['True', 'False'], value=str(param['default']), inline=True),
            ], className="p-2"))

    return children




@app.callback(
    Output('dataset-params-div', 'children'),
    Input('dataset-dropdown', 'value')
)
def update_dataset_params(selected_dataset):
    return create_dataset_params(selected_dataset)


@app.callback(
    Output('model-params-div', 'children'),
    Input('model-dropdown', 'value')
)
def update_model_params(selected_model):
    return create_model_params(selected_model)

@app.callback(
    Output("click-count", "children"),
    Output("progress-bar", "style", allow_duplicate=True),
    Input("update-button", "n_clicks"),
    State("click-count", "children"),
    prevent_initial_call=True
)
def update_click_count_and_show_progress(n_clicks, click_count):
    if n_clicks is None:
        raise PreventUpdate

    # Increment click count
    new_click_count = str(int(click_count) + 1)

    # Show progress bar
    progress_style = {"height": "20px", "visibility": "visible"}

    return new_click_count, progress_style



@callback(
    [
        Output("eig-plot", "figure"),
        Output("freq-plot", "figure"),
        Output("modes-plot", "figure"),
        Output("dataset-plot", "figure"),
        Output("T", "max"),
        Output("freq_range_slider", "max"),
        Output("freq_range_slider", "marks"),
        Output("modes_select", "options"),
        Output("rank-input", "value"),
        Output("tikhonov_reg-input", "value"),
        Output("context_window_len-input", "value"),
        Output("progress-bar", "style"),
    ],  

    [Input("click-count", "children"),],

    [
        State("freq_range_slider", "value"),
        State("Tmax", "value"),
        State("T", "value"),
        State("modes_select", "value"),
        State("rank-input", "value"),
        State("tikhonov_reg-input", "value"),
        State({'type': 'model-param', 'index': ALL}, 'value'),
        State("context_window_len-input", "value"),
        State({'type': 'dynamic-param', 'index': ALL}, 'value'),
        State("T_sample-input", "value"),
        State("dataset-dropdown", "value"),
        State("model-dropdown", "value"),
    ],

)
def main(click_count, value, Tmax, T, mode_selection, rank, tikhonov_reg, model_dynamic_params,
                       context_window_len, dynamic_params, T_sample, 
                       selected_dataset="LinearModel", selected_model="KernelDMD"):

    if click_count == "0":
        # Prevent update before the app loads (no button click yet)
        raise PreventUpdate

    print(dynamic_params)
    if selected_dataset == "LinearModel":
        # np.random.seed(10)
        # H = ortho_group.rvs(10)
        # eigs = np.exp(-np.arange(10))
        # A = H @ (eigs * np.eye(10)) @ H.T
        noise=1.
        rng_seed=None

        num_features=10
        r=5
        l=1

        if dynamic_params!=[]:
            noise = dynamic_params[0]
            rng_seed = dynamic_params[1]

            num_features = dynamic_params[2]
            r = dynamic_params[3]
            l = dynamic_params[4]
        
        if rng_seed is not None:
            np.random.seed(rng_seed)

        f = lambda x, r : 1. if x<r else 0.
        H = ortho_group.rvs(num_features) 
        eigs = np.array([l*f(i, r) for i in range(num_features)]) 
        A = H @ (eigs * np.eye(num_features)) @ H.T 

        dataset = LinearModel(A = A, noise=noise, rng_seed=rng_seed)  # Replace with the actual class and parameters
        # dataset.generate(X0 = np.zeros(A.shape[0]), T=100, rng_seed)
        _Z = dataset.sample(X0 = np.zeros(A.shape[0]), T = T_sample)
        #PLOT(_Z)
        X = traj_to_contexts(_Z, context_window_len=context_window_len)

    elif selected_dataset == "LogisticMap":
        r_param=4.0
        N_param=None
        rng_seed=None
        # time.sleep(0.05)
        if dynamic_params!=[]:
            r_param = dynamic_params[0]
            N_param = dynamic_params[1]
            rng_seed = dynamic_params[2]
        dataset = LogisticMap(r=r_param, N=N_param, rng_seed=rng_seed)  # Replace with the actual class and parameters
        _Z = dataset.sample(0.2, T_sample)
        X = traj_to_contexts(_Z,  context_window_len=context_window_len)

    elif selected_dataset == "LangevinTripleWell1D":
        gamma=0.1
        kt=1.0
        dt=1e-4
        rng_seed=None
        # time.sleep(0.05)
        if dynamic_params!=[]:
            gamma = dynamic_params[0]
            kt = dynamic_params[1]
            dt = dynamic_params[2]
            rng_seed = dynamic_params[3]
        dataset = LangevinTripleWell1D(gamma=gamma, kt=kt, dt=dt, rng_seed=rng_seed)
        _Z = dataset.sample(0., T_sample)
        X = traj_to_contexts(_Z,  context_window_len=context_window_len)

    elif selected_dataset == "DuffingOscillator":
        alpha=0.5
        beta=0.0625
        gamma=0.1
        delta=2.5
        omega=2.0
        dt=0.01
        # time.sleep(0.05)
        if dynamic_params!=[]:
            alpha = dynamic_params[0]
            beta = dynamic_params[1]
            gamma = dynamic_params[2]
            delta = dynamic_params[3]
            omega = dynamic_params[4]
            dt = dynamic_params[5]
        dataset = DuffingOscillator(alpha=alpha, beta=beta, gamma=gamma, delta=delta, omega=omega, dt=dt)
        _Z = dataset.sample(np.array([0.,0.]), T_sample)
        X = traj_to_contexts(_Z,  context_window_len=context_window_len)

    elif selected_dataset == "Lorenz63":
        sigma=10
        mu=28
        beta=8 / 3
        dt=0.01
        # time.sleep(0.05)
        if dynamic_params!=[]:
            sigma = dynamic_params[0]
            mu = dynamic_params[1]
            beta = dynamic_params[2]
            dt = dynamic_params[3]
        dataset = Lorenz63(sigma=sigma, mu=mu, beta=beta, dt=dt)
        _Z = dataset.sample(np.array([0,0.1,0]), T_sample)    
        X = traj_to_contexts(_Z,  context_window_len=context_window_len)

    fig_dataset = px.line(_Z, labels={"index":"time"})

    # print(X.shape)
    # print(X.shape[-1])
    #selected model
    print(model_dynamic_params)

    #Problem: les fonctions kernel ont besoin de parametres (par exemple: somme de x et y)
    if selected_model == "KernelDMD":
        # Assuming the order of parameters in create_model_params function
        operator_kwargs = {'kernel': DotProduct()}
        # time.sleep(0.05)
        if dynamic_params!=[]:
            kernel_mapping = {
            'DotProduct': DotProduct(),
            'RBF': RBF(length_scale=model_dynamic_params[4]), 
            'Matern': Matern(length_scale=model_dynamic_params[4]), 
            '0.5*DotProduct + 0.5*RBF': 0.5*DotProduct() + 0.5*RBF(length_scale=model_dynamic_params[4])
        }
            operator_kwargs = {
                'kernel': kernel_mapping.get(model_dynamic_params[3], DotProduct()), #DotProduct(),
                'reduced_rank': model_dynamic_params[0],
                # 'rank': model_params[2],
                # 'tikhonov_reg': model_params[3] if model_params[3] != '' else None,  # Handle empty string for None
                'svd_solver': model_dynamic_params[1],
                'iterated_power': model_dynamic_params[2],
                # 'n_oversamples': model_dynamic_params[3],
                # 'optimal_sketching': model_dynamic_params[5],
                'rng_seed': model_dynamic_params[5] if model_dynamic_params[5] != '' else None  # Handle empty string for None
            }
            print(operator_kwargs)
        # if rank is not None:
        operator_kwargs["rank"] = rank
        if tikhonov_reg is not None:
            operator_kwargs["tikhonov_reg"] = tikhonov_reg                        
        operator = KernelDMD(**operator_kwargs)

    #Problem: the feature_map is generated using a predefined dataset (LinearModel), do we need to fit the model on the same dataset?
    elif selected_model == "DeepEDMD":
        # time.sleep(0.05)
        operator_kwargs = {'feature_map': feature_map(data=X, context_window_len=context_window_len)}
        if dynamic_params!=[]:
            operator_kwargs = {
                'feature_map': feature_map(data=X, context_window_len=context_window_len, max_epochs=model_dynamic_params[0]),
                'reduced_rank': model_dynamic_params[1],
                'svd_solver': model_dynamic_params[2],
                'iterated_power': model_dynamic_params[3],
                # 'n_oversamples': model_dynamic_params[4],
                'rng_seed': model_dynamic_params[4] if model_dynamic_params[4] != '' else None  # Handle empty string for None
            }
        # if rank is not None:
        operator_kwargs["rank"] = rank
        if tikhonov_reg is not None:
            operator_kwargs["tikhonov_reg"] = tikhonov_reg    
        operator = DeepEDMD(**operator_kwargs)

    #Problem: do we need to consider other feature_maps ?
    elif selected_model == "ExtendedDMD":
        FeatureMap_mapping = {
            'IdentityFeatureMap': IdentityFeatureMap(),
        }
        operator_kwargs = {}
        # time.sleep(0.05)
        if dynamic_params!=[]:
            operator_kwargs = {
                'feature_map': FeatureMap_mapping.get(model_dynamic_params[3], IdentityFeatureMap()), #IdentityFeatureMap(),
                'reduced_rank': model_dynamic_params[0],
                'svd_solver': model_dynamic_params[1],
                'iterated_power': model_dynamic_params[2],
                # 'n_oversamples': model_dynamic_params[3],
                'rng_seed': model_dynamic_params[4] if model_dynamic_params[4] != '' else None  # Handle empty string for None
            }
        # if rank is not None:
        operator_kwargs["rank"] = rank
        if tikhonov_reg is not None:
            operator_kwargs["tikhonov_reg"] = tikhonov_reg    
        operator = ExtendedDMD(**operator_kwargs)

    elif selected_model == "DMD":
        operator_kwargs = {}
        # time.sleep(0.05)
        if dynamic_params!=[]:
            operator_kwargs = {
                'reduced_rank': model_dynamic_params[0],
                'svd_solver': model_dynamic_params[1],
                'iterated_power': model_dynamic_params[2],
                # 'n_oversamples': model_dynamic_params[3],
                'rng_seed': model_dynamic_params[3] if model_dynamic_params[3] != '' else None  # Handle empty string for None
            }
        # if rank is not None:
        operator_kwargs["rank"] = rank
        if tikhonov_reg is not None:
            operator_kwargs["tikhonov_reg"] = tikhonov_reg
        operator = DMD(**operator_kwargs)

    # operator = KernelDMD(DotProduct(), rank=10)
    operator.fit(X)
    viz = Visualizer(koopman=operator)
    available_modes = viz.infos["eig_num"].unique().astype("str")
    available_modes = np.insert(available_modes, 0, "All")
    available_modes = np.insert(available_modes, 1, "Combined")

    frequencies = viz.infos["frequency"].unique()
    pos_frequencies = frequencies[frequencies > 0]
    frequency_dict = {i: str(round(i, 3)) for i in pos_frequencies}
    frequency_dict[0] = "0"

    # Update the modes_select dropdown options
    modes_select_options = [{"label": str(i), "value": str(i)} for i in available_modes]

    if value is None:
        min_freq = viz.infos["frequency"].unique().min()
        max_freq = viz.infos["frequency"].unique().max()
    else:
        min_freq = value[0]
        max_freq = value[1]

    fig_eig = viz.plot_eigs(min_freq, max_freq)
    # print(viz) 
    fig_freqs = viz.plot_freqs(min_freq, max_freq)       

    if T is None:
        T = 1

    if mode_selection == "All":
        fig_modes = viz.plot_modes(index=None, min_freq=min_freq, max_freq=max_freq)
    elif mode_selection == "Combined":
        fig_modes = viz.plot_combined_modes(T, min_freq, max_freq)
    else:
        fig_modes = viz.plot_modes(
            index=[int(mode_selection)], min_freq=min_freq, max_freq=max_freq
        )
    # fig_pred = viz.plot_preds(operator.X_fit_[-1], 1, min_freq, max_freq)

    progress_style = {"height": "20px", "visibility": "hidden"}

    return (fig_eig, fig_freqs, fig_modes, fig_dataset, Tmax, 
            int(viz.infos["frequency"].max()) + 1, frequency_dict, 
            modes_select_options, rank, tikhonov_reg, context_window_len, progress_style)


app.run_server(debug=True)
