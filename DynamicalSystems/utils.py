import numpy as np
import matplotlib
import matplotlib.pyplot as plt

__useTeX__ = True
if __useTeX__:
    plt.rcParams.update({
        "text.usetex": True,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"]
        #"font.family": "sans-serif",
        #"font.sans-serif": ["Computer Modern Serif"]
    })

def plot_eigs(
        eigs,
        show_axes=True,
        figsize=(8, 8),
        title="",
        dpi=None,
        filename=None,
    ):
        ## Adapted from package pyDMD
        if dpi is not None:
            plt.figure(figsize=figsize, dpi=dpi)
        else:
            plt.figure(figsize=figsize)

        plt.title(title)
        plt.gcf()
        ax = plt.gca()

        (points,) = ax.plot(
            eigs.real, eigs.imag, "r+", label="Eigenvalues"
        )
        lim = 1.1
        supx, infx, supy, infy = lim, -lim, lim, -lim
        
        # set limits for axis
        ax.set_xlim((infx, supx))
        ax.set_ylim((infy, supy))
        

        plt.ylabel("Imaginary part")
        plt.xlabel("Real part")
        
        unit_circle = plt.Circle(
            (0.0, 0.0),
            1.0,
            color="k",
            fill=False,
            label="Unit circle",
            linestyle="-",
        )
        ax.add_artist(unit_circle)

        # Dashed grid
        gridlines = ax.get_xgridlines() + ax.get_ygridlines()
        for line in gridlines:
            line.set_linestyle("--")
        ax.grid(True)

        ax.add_artist(plt.legend([points], ["Eigenvalues"], loc="best", frameon=False))
        ax.set_aspect("equal")

        if filename:
            plt.savefig(filename)
        else:
            plt.show()