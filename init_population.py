import numpy as np
# Function to initialize a population for optimization algorithms
def initialize_population(pop_size=50, dimensions=2, bounds=(-5.12, 5.12), seed=42):
    np.random.seed(seed)  # Set random seed for reproducibility (same results each run)
    # Generate a 2D array of shape (pop_size, dimensions)
    # Each value is a random float uniformly sampled from the range [bounds[0], bounds[1]]
    return np.random.uniform(bounds[0], bounds[1], size=(pop_size, dimensions))