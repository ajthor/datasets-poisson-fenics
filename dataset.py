"""
2D Poisson equation with Dirichlet boundary conditions.

Solves: -∇²u = f
where u(x,y) is the solution field and f(x,y) is the source term.

Physical system: Steady-state diffusion in a 2D rectangular domain with
zero boundary conditions and random source distributions.
"""

import numpy as np
from torch.utils.data import IterableDataset

import fenics as fe
from functools import partial
from sklearn.metrics.pairwise import rbf_kernel
import logging

logger = logging.getLogger(__name__)


def sample_gp_prior(kernel, X, n_samples=1):
    """
    Sample from Gaussian Process prior - KEEP if using GP-based initial conditions.
    
    INSTRUCTIONS FOR CLAUDE:
    - This function samples from GP prior (zero mean) for random smooth fields
    - REMOVE this function if your dataset doesn't use GP sampling  
    - KEEP for generating smooth random initial conditions
    - Requires a kernel function (e.g., from sklearn.metrics.pairwise.rbf_kernel)
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    K = kernel(X, X)

    prior = np.random.multivariate_normal(
        mean=np.zeros(X.shape[0]),
        cov=K,
        size=n_samples,
    )

    return prior


def sample_gp_posterior(kernel, X, y, xt, n_samples=1):
    """
    Sample from Gaussian Process posterior - KEEP if using GP-based initial conditions.

    INSTRUCTIONS FOR CLAUDE:
    - This function is commonly used across datasets for smooth random initial conditions
    - REMOVE this function if your dataset doesn't use GP sampling
    - KEEP if you want smooth, correlated random fields as initial conditions
    - Requires sklearn.metrics.pairwise.rbf_kernel import (or similar kernel function)
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if xt.ndim == 1:
        xt = xt.reshape(-1, 1)

    K = kernel(X, X)
    Kt = kernel(X, xt)
    Ktt = kernel(xt, xt)

    K_inv = np.linalg.inv(K)

    mu = Kt.T @ K_inv @ y
    cov = Ktt - Kt.T @ K_inv @ Kt

    mu = mu.squeeze()

    posterior = np.random.multivariate_normal(
        mean=mu,
        cov=cov,
        size=n_samples,
    )

    return posterior


# TODO: Add additional utility functions as needed for your specific PDE
# Common examples:
# - Vortex generation functions for fluid dynamics
# - Wave packet generators for wave equations
# - Heat source/sink generators for thermal problems


class PoissonDataset(IterableDataset):
    """
    Dataset for 2D Poisson equation simulations with Dirichlet boundary conditions.
    Solves: -∇²u = f with u=0 on boundaries
    """
    def __init__(
        self,
        # Domain parameters
        Lx=1.0,                   # Domain width
        Ly=1.0,                   # Domain height
        Nx=32,                    # Grid points in x
        Ny=32,                    # Grid points in y
        # PDE parameters
        source_strength=1.0,      # Source term strength
        dtype=np.float64,
    ):
        """
        Dataset for 2D Poisson equation simulations with Dirichlet boundary conditions.
        Solves: -∇²u = f
        
        Args:
            Lx: Domain width in x-direction
            Ly: Domain height in y-direction
            Nx: Number of grid points in x-direction
            Ny: Number of grid points in y-direction
            source_strength: Source term strength
            dtype: Data type for computations
        """
        super().__init__()
        
        # Store domain and grid parameters
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        
        # Store PDE parameters
        self.source_strength = source_strength
        self.dtype = dtype
        
        # Setup FEniCS mesh and function space
        self.mesh = fe.RectangleMesh(fe.Point(0, 0), fe.Point(Lx, Ly), Nx-1, Ny-1)
        self.V = fe.FunctionSpace(self.mesh, 'P', 1)
        
        # Define boundary condition (u = 0 on boundary)
        def boundary(x, on_boundary):
            return on_boundary
        
        self.bc = fe.DirichletBC(self.V, fe.Constant(0), boundary)
        
        # Get mesh coordinates
        self.coordinates = self.V.tabulate_dof_coordinates()

    def __iter__(self):
        """
        Generate infinite samples from the dataset.
        
        INSTRUCTIONS FOR CLAUDE:
        - KEEP this method signature and infinite loop structure
        - Replace the initial condition generation with your method
        - Always end with: yield self.solve(initial_condition)
        """
        while True:
            # Generate random source term using GP
            sigma = 0.2 * min(self.Lx, self.Ly)
            gamma = 1 / (2 * sigma**2)
            
            # Create GP-based smooth source term
            source_amplitude = np.random.uniform(0.5, 2.0) * self.source_strength
            f_source = sample_gp_prior(
                kernel=partial(rbf_kernel, gamma=gamma),
                X=self.coordinates,
                n_samples=1,
            )[0] * source_amplitude
            
            # Solve the PDE and yield result
            yield self.solve(f_source)

    def solve(self, source_term):
        """
        Solve the Poisson equation for a given source term.

        Args:
            source_term: Source term f as a numpy array.

        Returns:
            A dictionary containing all data useful for learning the PDE.
        """
        
        # Set up FEniCS functions
        u = fe.Function(self.V)
        v = fe.TestFunction(self.V)
        
        # Create source function from array
        f = fe.Function(self.V)
        f.vector()[:] = source_term
        
        # Define variational problem: -∇²u = f
        # Weak form: ∫ ∇u·∇v dx = ∫ f·v dx
        a = fe.dot(fe.grad(u), fe.grad(v)) * fe.dx
        L = f * v * fe.dx
        
        # Solve the linear system
        fe.solve(a == L, u, self.bc)
        
        # Get solution as array
        u_solution = u.vector().get_local()

        return {
            # Coordinates
            "spatial_coordinates": self.coordinates,  # Shape: (N, 2) for 2D
            
            # Solution and source
            "u_solution": u_solution,
            "source_term": source_term,
            
            # PDE parameters
            "source_strength": self.source_strength,
            "domain_size": [self.Lx, self.Ly],
        }
