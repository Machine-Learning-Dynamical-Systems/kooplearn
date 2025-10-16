"""
Spectral decomposition solver for Fokker-Planck operator
associated with overdamped Langevin dynamics in 1D and 2D.

The operator L = -∇·(∇V ·) + Δ is discretized using cosine basis functions
with Galerkin projection.
"""

import numpy as np
from scipy.linalg import eig

# ============================================================================
# 1D Basis Functions and Quadrature
# ============================================================================


def build_cosine_basis_1d(domain_min, domain_max, num_basis, num_quad_points=None):
    """
    Construct 1D cosine basis functions and derivatives on quadrature grid.

    Args:
        domain_min (float): Left boundary of domain [a, b]
        domain_max (float): Right boundary of domain
        num_basis (int): Number of cosine basis functions
        num_quad_points (int, optional): Number of quadrature points.
                                         Defaults to 2 * num_basis

    Returns:
        quad_points (ndarray): Quadrature points, shape (num_quad_points,)
        quad_weights (ndarray): Trapezoidal quadrature weights
        basis_vals (ndarray): Basis function values, shape (num_basis, num_quad_points)
        basis_derivs (ndarray): Basis function derivatives, shape (num_basis, num_quad_points)
    """
    if num_quad_points is None:
        num_quad_points = 2 * num_basis

    # Quadrature grid with trapezoidal weights
    quad_points = np.linspace(domain_min, domain_max, num_quad_points)
    domain_length = domain_max - domain_min
    quad_weights = np.ones(num_quad_points) * domain_length / (num_quad_points - 1)
    quad_weights[0] /= 2
    quad_weights[-1] /= 2

    # Scaled coordinates in [0, 1]
    scaled_coords = (quad_points - domain_min) / domain_length

    # Cosine basis: φ_k(x) = cos(kπ(x-a)/(b-a))
    mode_indices = np.arange(num_basis)
    basis_vals = np.cos(np.pi * np.outer(mode_indices, scaled_coords))

    # Derivatives: φ'_k(x) = -kπ/(b-a) sin(kπ(x-a)/(b-a))
    basis_derivs = (-np.pi * mode_indices[:, None] / domain_length) * np.sin(
        np.pi * np.outer(mode_indices, scaled_coords)
    )

    return quad_points, quad_weights, basis_vals, basis_derivs


# ============================================================================
# 1D Matrix Assembly
# ============================================================================


def assemble_operators_1d(
    domain_min,
    domain_max,
    num_basis,
    potential_gradient,
    gamma,
    sigma,
    num_quad_points=None,
):
    """
    Assemble 1D Fokker-Planck operator matrices using Galerkin projection.

    The operator is L = -d/dx(V'(x) ·) - d²/dx²

    Args:
        domain_min, domain_max (float): Domain boundaries [a, b]
        num_basis (int): Number of cosine basis functions
        potential_gradient (callable): Function V'(x) returning gradient at points x
        num_quad_points (int, optional): Number of quadrature points

    Returns:
        stiffness_matrix (ndarray): Operator matrix A, shape (num_basis, num_basis)
        mass_matrix (ndarray): Mass matrix B, shape (num_basis, num_basis)
    """
    quad_points, quad_weights, basis_vals, basis_derivs = build_cosine_basis_1d(
        domain_min, domain_max, num_basis, num_quad_points
    )

    # Evaluate potential gradient at quadrature points
    gradient_vals = potential_gradient(quad_points)

    # Mass matrix: B_ij = ∫ φ_i φ_j dx
    mass_matrix = np.tensordot(basis_vals, basis_vals * quad_weights, axes=(1, 1))

    # Diffusion (stiffness): -∫ φ'_i φ'_j dx
    diffusion_matrix = -np.tensordot(
        basis_derivs, basis_derivs * quad_weights, axes=(1, 1)
    )

    # Advection: -∫ φ_i V'(x) φ'_j dx
    advection_matrix = -np.tensordot(
        basis_vals, (gradient_vals[None, :] * basis_derivs) * quad_weights, axes=(1, 1)
    )

    stiffness_matrix = (
        sigma * sigma * diffusion_matrix / (2 * gamma) + advection_matrix
    ) / gamma

    return stiffness_matrix, mass_matrix


# ============================================================================
# 2D Matrix Assembly
# ============================================================================


def assemble_operators_2d(
    domain_bounds,
    num_basis,
    potential_gradient,
    gamma,
    sigma,
    num_quad_points=None,
    method="vectorized",
):
    """
    Assemble 2D Fokker-Planck operator using tensor product cosine basis.

    The operator is L = -∇·(∇V ·) - Δ

    Args:
        domain_bounds (tuple): (x_min, x_max, y_min, y_max)
        num_basis (tuple): (num_basis_x, num_basis_y) number of modes per dimension
        potential_gradient (callable): Function returning (grad_x, grad_y)
                                       given arrays X, Y of shape (Mx, My)
        num_quad_points (tuple, optional): (num_quad_x, num_quad_y)
        method (str): Assembly method - 'vectorized' (fast, more memory) or
                     'kronecker' (slower, less memory). Default: 'vectorized'

    Returns:
        stiffness_matrix (ndarray): Operator matrix A, shape (N, N) where N = Nx*Ny
        mass_matrix (ndarray): Mass matrix B, shape (N, N)
    """
    x_min, x_max, y_min, y_max = domain_bounds
    num_basis_x, num_basis_y = num_basis

    if num_quad_points is None:
        num_quad_points = (2 * num_basis_x, 2 * num_basis_y)
    num_quad_x, num_quad_y = num_quad_points

    # 1D quadrature rules
    x_quad = np.linspace(x_min, x_max, num_quad_x)
    y_quad = np.linspace(y_min, y_max, num_quad_y)

    weights_x = np.ones(num_quad_x) * (x_max - x_min) / (num_quad_x - 1)
    weights_x[0] /= 2
    weights_x[-1] /= 2

    weights_y = np.ones(num_quad_y) * (y_max - y_min) / (num_quad_y - 1)
    weights_y[0] /= 2
    weights_y[-1] /= 2

    # 2D quadrature grid
    X_grid, Y_grid = np.meshgrid(x_quad, y_quad, indexing="ij")
    weights_2d = weights_x[:, None] * weights_y[None, :]

    # Evaluate potential gradient on grid
    grad_x, grad_y = potential_gradient(X_grid, Y_grid)

    # Build 1D cosine basis functions and derivatives
    mode_idx_x = np.arange(num_basis_x).reshape(-1, 1)
    mode_idx_y = np.arange(num_basis_y).reshape(-1, 1)

    # Basis functions
    x_scaled = (x_quad - x_min) / (x_max - x_min)
    y_scaled = (y_quad - y_min) / (y_max - y_min)

    basis_x = np.cos(mode_idx_x * np.pi * x_scaled)  # (Nx, num_quad_x)
    basis_y = np.cos(mode_idx_y * np.pi * y_scaled)  # (Ny, num_quad_y)

    # Derivatives
    deriv_x = (
        -mode_idx_x * np.pi / (x_max - x_min) * np.sin(mode_idx_x * np.pi * x_scaled)
    )
    deriv_y = (
        -mode_idx_y * np.pi / (y_max - y_min) * np.sin(mode_idx_y * np.pi * y_scaled)
    )

    # Choose assembly method
    if method == "kronecker":
        # Memory-efficient Kronecker product approach (less memory, slightly slower)
        stiffness_matrix, mass_matrix = _assemble_2d_kronecker(
            basis_x,
            basis_y,
            deriv_x,
            deriv_y,
            weights_x,
            weights_y,
            grad_x,
            grad_y,
            gamma,
            sigma,
        )
    else:  # method == 'vectorized'
        # Fast vectorized approach (more memory, much faster)
        stiffness_matrix, mass_matrix = _assemble_2d_vectorized(
            num_basis_x,
            num_basis_y,
            num_quad_x,
            num_quad_y,
            basis_x,
            basis_y,
            deriv_x,
            deriv_y,
            weights_2d,
            grad_x,
            grad_y,
            gamma,
            sigma,
        )
    return stiffness_matrix, mass_matrix


def _assemble_2d_vectorized(
    num_basis_x,
    num_basis_y,
    num_quad_x,
    num_quad_y,
    basis_x,
    basis_y,
    deriv_x,
    deriv_y,
    weights_2d,
    grad_x,
    grad_y,
    gamma,
    sigma,
):
    """
    Vectorized assembly using einsum - fastest but uses more memory.
    """
    total_basis = num_basis_x * num_basis_y

    # Precompute tensor product basis functions
    basis_2d = np.zeros((total_basis, num_quad_x, num_quad_y))
    deriv_x_2d = np.zeros((total_basis, num_quad_x, num_quad_y))
    deriv_y_2d = np.zeros((total_basis, num_quad_x, num_quad_y))

    idx = 0
    for i in range(num_basis_x):
        for j in range(num_basis_y):
            basis_2d[idx] = np.outer(basis_x[i], basis_y[j])
            deriv_x_2d[idx] = np.outer(deriv_x[i], basis_y[j])
            deriv_y_2d[idx] = np.outer(basis_x[i], deriv_y[j])
            idx += 1

    # Vectorized assembly using Einstein summation
    # Apply quadrature weights to basis functions once
    basis_weighted = basis_2d * weights_2d[None, :, :]
    deriv_x_weighted = deriv_x_2d * weights_2d[None, :, :]
    deriv_y_weighted = deriv_y_2d * weights_2d[None, :, :]

    # Mass matrix: B[i,j] = ∫∫ φ_i φ_j dx dy
    # Using einsum: sum over spatial dimensions (axes 1,2)
    mass_matrix = np.einsum("ixy,jxy->ij", basis_2d, basis_weighted)

    # Diffusion term: -∫∫ (∇φ_i · ∇φ_j) dx dy
    diffusion_matrix = -(
        np.einsum("ixy,jxy->ij", deriv_x_2d, deriv_x_weighted)
        + np.einsum("ixy,jxy->ij", deriv_y_2d, deriv_y_weighted)
    )

    # Advection term: -∫∫ φ_i (∇V · ∇φ_j) dx dy
    # Precompute drift-weighted derivatives
    drift_deriv_x = grad_x * deriv_x_2d  # (total_basis, Mx, My)
    drift_deriv_y = grad_y * deriv_y_2d  # (total_basis, Mx, My)

    advection_matrix = -np.einsum(
        "ixy,jxy->ij", basis_2d, (drift_deriv_x + drift_deriv_y) * weights_2d
    )

    stiffness_matrix = (
        sigma * sigma * diffusion_matrix / (2 * gamma) + advection_matrix
    ) / gamma

    return stiffness_matrix, mass_matrix


def _assemble_2d_kronecker(
    basis_x,
    basis_y,
    deriv_x,
    deriv_y,
    weights_x,
    weights_y,
    grad_x,
    grad_y,
    gamma,
    sigma,
):
    """
    Memory-efficient assembly using Kronecker products and 1D integrals.
    Works when drift is separable or can be approximated along coordinate axes.
    """
    # 1D mass and stiffness matrices
    mass_x = basis_x @ np.diag(weights_x) @ basis_x.T
    mass_y = basis_y @ np.diag(weights_y) @ basis_y.T
    stiff_x = deriv_x @ np.diag(weights_x) @ deriv_x.T
    stiff_y = deriv_y @ np.diag(weights_y) @ deriv_y.T

    # Laplacian via Kronecker sum: -(A_x ⊗ M_y + M_x ⊗ A_y)
    diffusion_matrix = -(np.kron(stiff_x, mass_y) + np.kron(mass_x, stiff_y))

    # For advection, we need cross-terms which require 2D integration
    # This is an approximation using coordinate-wise drift averages
    drift_x_avg = np.mean(grad_x, axis=1)  # Average over y
    drift_y_avg = np.mean(grad_y, axis=0)  # Average over x

    # Advection matrices: -∫ φ_i V'(x) φ'_j dx (treating drift as separable)
    adv_x = -basis_x @ np.diag(weights_x * drift_x_avg) @ deriv_x.T
    adv_y = -basis_y @ np.diag(weights_y * drift_y_avg) @ deriv_y.T

    advection_matrix = np.kron(adv_x, mass_y) + np.kron(mass_x, adv_y)

    stiffness_matrix = (
        sigma * sigma * diffusion_matrix / (2 * gamma) + advection_matrix
    ) / gamma
    mass_matrix = np.kron(mass_x, mass_y)

    return stiffness_matrix, mass_matrix


# ============================================================================
# Eigenfunction Evaluation
# ============================================================================


def eval_eigenfunctions_1d(eigenvectors, eval_points, domain_min, domain_max):
    """
    Evaluate 1D eigenfunctions from cosine basis expansion coefficients.

    Args:
        eigenvectors (ndarray): (num_basis, num_modes) matrix of eigenvector coefficients
        eval_points (ndarray): Points where to evaluate the eigenfunctions
        domain_min, domain_max (float): Domain boundaries [a, b]

    Returns:
        eigenfunctions (ndarray): Shape (len(eval_points), num_modes)
                                  eigenfunction values at evaluation points
    """
    num_basis = eigenvectors.shape[0]
    mode_indices = np.arange(num_basis)

    # Cosine basis evaluated at eval_points
    scaled_points = (eval_points - domain_min) / (domain_max - domain_min)
    basis_at_points = np.cos(np.pi * np.outer(mode_indices, scaled_points))

    # Linear combination: u_j(x) = Σ_k c^(j)_k φ_k(x)
    eigenfunctions = basis_at_points.T @ eigenvectors

    return eigenfunctions


def eval_eigenfunctions_2d(eigenvectors, x_points, y_points, domain_bounds, num_basis):
    """
    Evaluate 2D eigenfunctions on a grid from cosine basis expansion coefficients.

    Args:
        eigenvectors (ndarray): (N, num_modes) eigenvector coefficients
                                where N = num_basis_x * num_basis_y (row-major flattening)
        x_points, y_points (ndarray): 1D coordinate arrays for evaluation grid
        domain_bounds (tuple): (x_min, x_max, y_min, y_max)
        num_basis (tuple): (num_basis_x, num_basis_y) number of modes per dimension

    Returns:
        eigenfunctions (ndarray): Shape (len(x_points), len(y_points), num_modes)
                                  eigenfunction values on the 2D grid
    """
    x_min, x_max, y_min, y_max = domain_bounds
    num_basis_x, num_basis_y = num_basis
    num_modes = eigenvectors.shape[1]

    # Scaled coordinates
    x_scaled = (x_points - x_min) / (x_max - x_min)
    y_scaled = (y_points - y_min) / (y_max - y_min)

    # 1D cosine basis matrices
    mode_idx_x = np.arange(num_basis_x).reshape(-1, 1)
    mode_idx_y = np.arange(num_basis_y).reshape(-1, 1)

    basis_x = np.cos(mode_idx_x * np.pi * x_scaled[None, :])  # (Nx, len(x_points))
    basis_y = np.cos(mode_idx_y * np.pi * y_scaled[None, :])  # (Ny, len(y_points))

    # Allocate output array
    eigenfunctions = np.zeros((len(x_points), len(y_points), num_modes))

    # Reconstruct each eigenfunction via tensor contraction
    for mode_idx in range(num_modes):
        # Reshape flattened coefficients to 2D grid
        coeff_grid = eigenvectors[:, mode_idx].reshape(num_basis_x, num_basis_y)

        # Tensor product: u(x,y) = Σ_{i,j} c_{ij} φ_i(x) φ_j(y)
        # Implemented as: basis_x.T @ coeff_grid @ basis_y
        eigenfunctions[:, :, mode_idx] = basis_x.T @ coeff_grid @ basis_y

    return eigenfunctions


# ============================================================================
# Utility Functions
# ============================================================================


def sort_indices_by_magnitude(vector):
    """
    Return indices that would sort vector by absolute value (ascending).

    Args:
        vector (ndarray): Input vector

    Returns:
        sort_indices (ndarray): Permutation indices
    """
    abs_values = np.abs(vector)
    sort_indices = np.argsort(abs_values)
    return sort_indices


def compute_prinz_potential_eig(gamma, sigma, dt, eval_right_on, num_components=4):
    prinz_grad = lambda x: (
        -128 * np.exp(-80 * ((-0.5 + x) ** 2)) * (-0.5 + x)
        - 512 * np.exp(-80 * (x**2)) * x
        + 32 * (x**7)
        - 160 * np.exp(-40 * ((0.5 + x) ** 2)) * (0.5 + x)
    )
    Nx = 128
    domain = (-3, 3)  # A good value for the prinz potential
    A, B = assemble_operators_1d(*domain, Nx, prinz_grad, gamma, sigma)

    eigvals, eigvecs = eig(A, B)
    sorted_idxs = sort_indices_by_magnitude(eigvals)
    eigvals = eigvals[sorted_idxs].real
    eigvecs = eigvecs[:, sorted_idxs].real

    u = eval_eigenfunctions_1d(eigvecs, eval_right_on, *domain)

    return np.exp(eigvals * dt)[:num_components], u[:, :num_components]
