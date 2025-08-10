"""
2D Poisson equation with Dirichlet boundary conditions.

Solves: -∇²u = f
where u(x,y) is the solution field and f(x,y) is the source term.

Physical system: Steady-state diffusion in a 2D rectangular domain with
zero boundary conditions and random source distributions.
"""

import numpy as np
from torch.utils.data import IterableDataset

import ufl
import numpy
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
from functools import partial
from sklearn.metrics.pairwise import rbf_kernel
import logging

logger = logging.getLogger(__name__)


def sample_gp_prior(kernel, X, n_samples=1):
    """
    Sample from Gaussian Process prior for random smooth fields.
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
    Sample from Gaussian Process posterior for smooth random fields.
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
        
        # Create DOLFINx mesh and function space
        self.domain = mesh.create_rectangle(
            MPI.COMM_WORLD, 
            [(0.0, 0.0), (Lx, Ly)], 
            [Nx-1, Ny-1],
            cell_type=mesh.CellType.triangle
        )
        self.V = fem.functionspace(self.domain, ("Lagrange", 1))
        
        # Get DOF coordinates
        self.coordinates = self.V.tabulate_dof_coordinates()

    def __iter__(self):
        """
        Generate infinite samples from the dataset.
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
        
        # Define trial and test functions
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        
        # Create source function from array
        f = fem.Function(self.V)
        f.x.array[:] = source_term
        
        # Define variational problem: -∇²u = f
        # Weak form: ∫ ∇u·∇v dx = ∫ f·v dx
        a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = f * v * ufl.dx
        
        # Apply homogeneous Dirichlet boundary conditions on all boundaries
        def boundary_marker(x):
            return numpy.full(x.shape[1], True, dtype=bool)
        
        fdim = self.domain.topology.dim - 1
        boundary_facets = mesh.locate_entities_boundary(self.domain, fdim, boundary_marker)
        boundary_dofs = fem.locate_dofs_topological(self.V, fdim, boundary_facets)
        bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, self.V)
        
        # Solve the linear problem
        problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = problem.solve()
        
        # Get solution as array
        u_solution = uh.x.array.copy()

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
