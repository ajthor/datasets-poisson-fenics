#!/usr/bin/env python3
"""
Plot a single sample from the 2D Poisson equation dataset.

Visualizes the source term and solution field for a single 
randomly generated Poisson boundary value problem.
"""

import numpy as np
import matplotlib.pyplot as plt
from poisson_dataset import PoissonDataset


def plot_poisson_sample(sample, save_path="sample_plot.png"):
    """
    Plot a single sample from the 2D Poisson equation dataset.

    Creates a 2-panel visualization showing:
    1. Source term f(x,y)  
    2. Solution field u(x,y)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Extract data from dataset return dictionary
    coordinates = sample["spatial_coordinates"]  # Shape: (N, 2)
    solution = sample["u_solution"]  # Shape: (N,)
    source = sample["source_term"]   # Shape: (N,)
    domain_size = sample["domain_size"]  # [Lx, Ly]
    
    # Create regular grid for imshow plotting
    from scipy.interpolate import griddata
    
    # Define grid resolution
    grid_size = 64
    x_grid = np.linspace(0, domain_size[0], grid_size)
    y_grid = np.linspace(0, domain_size[1], grid_size)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Interpolate scattered data onto regular grid
    coords_2d = coordinates[:, :2]  # Use only x,y coordinates
    source_grid = griddata(coords_2d, source, (X_grid, Y_grid), method='linear', fill_value=0)
    solution_grid = griddata(coords_2d, solution, (X_grid, Y_grid), method='linear', fill_value=0)
    
    # Plot 1: Source term f(x,y)
    im1 = ax1.imshow(source_grid, extent=[0, domain_size[0], 0, domain_size[1]], 
                     origin='lower', cmap="RdBu_r", aspect='equal')
    ax1.set_xlabel("x")
    ax1.set_ylabel("y") 
    ax1.set_title("Source Term f(x,y)")
    plt.colorbar(im1, ax=ax1, label="f(x,y)")
    
    # Plot 2: Solution field u(x,y)
    im2 = ax2.imshow(solution_grid, extent=[0, domain_size[0], 0, domain_size[1]], 
                     origin='lower', cmap="viridis", aspect='equal')
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Solution u(x,y)")
    plt.colorbar(im2, ax=ax2, label="u(x,y)")
    
    plt.suptitle("2D Poisson Equation: -∇²u = f", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"Sample visualization saved to {save_path}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create dataset instance
    dataset = PoissonDataset()
    
    # Generate a single sample
    dataset_iter = iter(dataset)
    sample = next(dataset_iter)
    
    print("Sample keys:", list(sample.keys()))
    for key, value in sample.items():
        if hasattr(value, "shape"):
            print(f"{key}: shape {value.shape}")
        else:
            print(f"{key}: {type(value)} - {value}")
    
    # Plot the sample
    plot_poisson_sample(sample)