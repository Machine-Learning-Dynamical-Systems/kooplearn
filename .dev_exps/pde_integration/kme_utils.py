from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from kooplearn.models.kernel import KernelLowRankRegressor

def evolve_density(
        t: int, 
        fitted_estimator: KernelLowRankRegressor,
        init_density: ArrayLike, 
        final_density_sampled_at: ArrayLike
        ) -> ArrayLike:
    Y_train = fitted_estimator.Y_fit_
    _kernel = fitted_estimator.kernel
    _out = _kernel(Y_train, final_density_sampled_at)
    return fitted_estimator.predict(init_density, t, observables=_out)

def make_movie(
        fitted_estimator: KernelLowRankRegressor,
        init_density: ArrayLike,
        target_density: ArrayLike,
        sample_points: ArrayLike,
        filename: str = 'pde_evolution.gif',
        duration_in_seconds: float = 5.0,
        fps: int = 30,
        dpi: int = 72,
        scale: float = 1.0
):
    _num_frames = int(duration_in_seconds * fps)

    _frames = np.empty((_num_frames, *sample_points.shape))
    
    for frame_idx in tqdm(range(_num_frames), desc='Evolving initial density', unit='frame', leave=False):
        _frames[frame_idx] = (scale**frame_idx)*evolve_density(frame_idx, fitted_estimator, init_density, sample_points)
        _frames[frame_idx] /= _frames[frame_idx].max()

    ax_settings = {
        'xlim': (-2, 2),
        'ylim': (0, 1.05)
    }

    def _update(frame_idx: int, ax: plt.Axes):
        frame = _frames[frame_idx]
        ax.cla()
        ax.plot(sample_points, frame, color='black')
        ax.plot(sample_points, target_density/target_density.max(), color='red')
        ax.set_title(f't = {frame_idx}')
        ax.set(**ax_settings)

    fig, ax = plt.subplots()
    ani = animation.FuncAnimation(fig, _update, frames=_num_frames, fargs=(ax,), interval=1000/fps)
    ani.save(filename, dpi=dpi, writer='pillow')