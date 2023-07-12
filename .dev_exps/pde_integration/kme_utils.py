from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from kooplearn.models.kernel import KernelLowRankRegressor

def evolve_density(
        t: float, 
        fitted_estimator: KernelLowRankRegressor,
        init_density: ArrayLike, 
        final_density_sampled_at: ArrayLike
        ) -> ArrayLike:
    w, left_fns, right_fns = fitted_estimator.eig(eval_left_on= final_density_sampled_at, eval_right_on= init_density)
    w = np.diag(w**t)
    return np.squeeze(np.linalg.multi_dot([left_fns, w, right_fns.T]))

def make_movie(
        fitted_estimator: KernelLowRankRegressor,
        init_density: ArrayLike,
        target_density: ArrayLike,
        sample_points: ArrayLike,
        filename: str = 'pde_evolution.gif',
        duration_in_seconds: float = 5.0,
        time_horizon: float = 2.0,
        fps: int = 24,
        dpi: int = 72,
        scale: float = 1.0
):
    _num_frames = int(duration_in_seconds * fps)
    _frames = np.empty((_num_frames, *sample_points.shape))
    _times = np.linspace(0, time_horizon, _num_frames)
    for frame_idx in tqdm(range(_num_frames), desc='Evolving initial density', unit='frame', leave=False):
        _frames[frame_idx] = (scale**frame_idx)*evolve_density(_times[frame_idx], fitted_estimator, init_density, sample_points)
        _frames[frame_idx] /= _frames[frame_idx].max()

    ax_settings = {
        'xlim': (-2, 2),
        'ylim': (-0.5, 1.05)
    }

    def _update(frame_idx: int, ax: plt.Axes):
        frame = _frames[frame_idx]
        ax.cla()
        ax.plot(sample_points, frame, color='black')
        ax.plot(sample_points, target_density/target_density.max(), color='red')
        ax.axhline(y = 0, linewidth=0.5, linestyle='--')
        ax.set_title(f't = {_times[frame_idx]:.1f}')
        ax.set(**ax_settings)

    fig, ax = plt.subplots()
    ani = animation.FuncAnimation(fig, _update, frames=_num_frames, fargs=(ax,), interval=1000/fps)
    ani.save(filename, dpi=dpi, writer='pillow')