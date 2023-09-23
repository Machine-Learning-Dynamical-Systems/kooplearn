def check_torch_deps():
    try:
        import lightning
    except ImportError:
        raise ImportError(
            "To use kooplearn's deep learning models please reinstall it with the `torch` extra flag by typing `pip install kooplearn[torch]`."
        )


def check_dashboard_deps():
    try:
        import dash_iconify
        import dash_mantine_components
        import plotly
    except ImportError:
        raise ImportError(
            "To use kooplearn's dashboard please reinstall it with the `dashboard` extra flag by typing `pip install kooplearn[dashboard]`."
        )
