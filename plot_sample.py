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
    
    # Create scatter plots since we have unstructured mesh data
    # Plot 1: Source term f(x,y)
    scatter1 = ax1.scatter(
        coordinates[:, 0], coordinates[:, 1], 
        c=source, cmap="RdBu_r", s=1, rasterized=True
    )
    ax1.set_xlabel("x")
    ax1.set_ylabel("y") 
    ax1.set_title("Source Term f(x,y)")
    ax1.set_aspect("equal")
    ax1.set_xlim(0, domain_size[0])
    ax1.set_ylim(0, domain_size[1])
    plt.colorbar(scatter1, ax=ax1, label="f(x,y)")
    
    # Plot 2: Solution field u(x,y)
    scatter2 = ax2.scatter(
        coordinates[:, 0], coordinates[:, 1],
        c=solution, cmap="viridis", s=1, rasterized=True
    )
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Solution u(x,y)")
    ax2.set_aspect("equal")
    ax2.set_xlim(0, domain_size[0])
    ax2.set_ylim(0, domain_size[1])
    plt.colorbar(scatter2, ax=ax2, label="u(x,y)")
    
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