import argparse
import pickle

import dash_mantine_components as dmc
from dash import Dash
from sklearn.gaussian_process.kernels import DotProduct

from kooplearn._src.dashboard.dash_components.callbacks import get_cb_functions
from kooplearn._src.dashboard.dash_components.layout import header, plots_components
from kooplearn._src.dashboard.visualizer import Visualizer
from kooplearn.datasets import Mock

APP_TITLE = "Koopman Modes Dashboard"
DEBUG = True

app = Dash(
    "Koopman Modes Dashboard",
    external_stylesheets=[
        # include google fonts
        "https://fonts.googleapis.com/css2?family=Inter:wght@100;200;300;400;500;900&display=swap"
    ],
)


def main():
    parser = argparse.ArgumentParser(
        description="A dashboard to explore the Koopman modes"
    )
    parser.add_argument(  # location of the koopman estimator
        "--koopman", type=str, default=""
    )

    args = parser.parse_args()
    if args.koopman == "":
        # Tutorial mode
        dataset = Mock(num_features=10, rng_seed=0)
        _Z = dataset.sample(None, 100)
        X, Y = _Z[:-1], _Z[1:]
        operator = KernelReducedRank(DotProduct(), rank=10)
        operator.fit(X, Y)
    else:
        with open(args.koopman, "rb") as file:
            operator = pickle.load(file)

    viz = Visualizer(operator)
    frequencies = viz.infos["frequency"].unique()
    frequencies = frequencies[frequencies > 0]
    app.layout = dmc.MantineProvider(
        theme={
            "fontFamily": "'Inter', sans-serif",
            "primaryColor": "indigo",
            "colorScheme": "dark",
            "components": {
                "Button": {"styles": {"root": {"fontWeight": 400}}},
                "Alert": {"styles": {"title": {"fontWeight": 500}}},
                "AvatarGroup": {"styles": {"truncated": {"fontWeight": 500}}},
            },
        },
        inherit=True,
        withGlobalStyles=True,
        withNormalizeCSS=True,
        children=[header, plots_components(frequencies)],
    )
    callback_fn = get_cb_functions(viz)

    app.run_server(debug=DEBUG)


if __name__ == "__main__":
    main()
