#Imports
#STDLIB
import json
from typing import TypeVar, Callable, Tuple

#ESSENTIALS
import jax
import jax.numpy as jnp
import einops

#MISC
import jax_md

T = TypeVar('T')

class BoltzmannDistribution():
    def __init__(self, potential: Callable, beta: float, domain: Tuple):
        self.potential = potential
        self.beta = beta
        self.domain = domain
    def pdf(self, x):
        np.exp(-pande_potential(x)/temperature)

with open("config.json", "r") as f:
    configs = json.load(f)

@jax.jit
def potential(x):
    """
    See Example 1 of "Modeling Molecular Kinetics with tICA and the Kernel Trick" 10.1021/ct5007357
    """
    return 4*(x**8+ 0.8*jnp.exp(-80*(x**2)) +  0.2*jnp.exp(-80*((x - 0.5)**2)) + 0.5*jnp.exp(-40*((x + 0.5)**2)))

grad_potential = jax.grad(potential)

displacement_fn, shift_fn = jax_md.space.free()
init_simulation, _evolve = jax_md.simulate.brownian(lambda x: jnp.sum(potential(x)), shift_fn, configs["time_step"], configs["temperature"])
evolve_fn = jax.jit(_evolve)

def simulate(num_steps: int, state: T) -> T:
    state = jax.lax.fori_loop(0, num_steps, evolve_fn, state)
    return state

def sample_markov(
    num_samples: int, 
    timesteps_between_samples: int = 1, 
    num_datasets: int = 1, 
    seed: int = 0
    ) -> jnp.ndarray:

    key = jax.random.PRNGKey(seed)
    key_brownian, key_init_pos = jax.random.split(key)
    _initial_positions = jax.random.uniform(key_init_pos, (num_datasets, 1), minval=-1, maxval=1)
    state = init_simulation(key_brownian, _initial_positions)
    positions = jnp.zeros((num_samples, num_datasets, 1))
    for i in range(num_samples):
        state = simulate(timesteps_between_samples, state)
        positions.at[i].set(state.position)
    return einops.rearrange(positions, "s d 1 -> d s") #[num_datasets, num_samples]

def sample_iid(
    num_samples: int, 
    timesteps_between_samples: int = 1, 
    num_datasets: int = 1, 
    seed: int = 0
    ) -> jnp.array:

    pass

def compute_eigenvalues():
    pass

def save_data():
    pass