import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # Add this line to support CSV export
from init_population import initialize_population  # Custom population initialization function

# Rastrigin function (benchmark function with many local minima, difficult for optimization)
def rastrigin(x):
    return 20 + x[0]**2 + x[1]**2 - 10 * (np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))

# Particle Swarm Optimization main function
def particle_swarm_optimization():
    pop_size = 50                     # Number of particles in the swarm
    dimensions = 2                    # Number of dimensions (2D optimization)
    bounds = (-5.12, 5.12)            # Search space boundaries
    generations = 100                 # Number of iterations (generations)

    # PSO parameters
    w = 0.7                           # Inertia weight (controls momentum)
    c1 = 1.5                          # Cognitive coefficient (self-exploration)
    c2 = 1.5                          # Social coefficient (swarm-exploration)

    # Initialize particle positions randomly within the bounds
    positions = initialize_population(pop_size, dimensions, bounds)

    # Initialize velocities of all particles to zero
    velocities = np.zeros_like(positions)

    # Initialize personal best positions (each particle's best known position)
    personal_best = positions.copy()

    # Evaluate fitness of personal bests
    personal_best_fitness = np.array([rastrigin(p) for p in positions])

    # Identify global best among all personal bests
    global_best = personal_best[np.argmin(personal_best_fitness)]

    # Lists to store best fitness and corresponding solution for each generation
    best_fitness_list = []
    best_solution_list = []

    # Main optimization loop
    for _ in range(generations):
        for i in range(pop_size):
            # Evaluate current position
            fitness = rastrigin(positions[i])

            # Update personal best if current is better
            if fitness < personal_best_fitness[i]:
                personal_best[i] = positions[i]
                personal_best_fitness[i] = fitness

        # Update global best based on all personal bests
        global_best = personal_best[np.argmin(personal_best_fitness)]

        for i in range(pop_size):
            # Generate random vectors for stochastic behavior
            r1 = np.random.rand(dimensions)
            r2 = np.random.rand(dimensions)

            # Update velocity using the PSO formula
            velocities[i] = (
                w * velocities[i] +
                c1 * r1 * (personal_best[i] - positions[i]) +
                c2 * r2 * (global_best - positions[i])
            )

            # Update position based on new velocity
            positions[i] += velocities[i]

            # Ensure the updated positions are within the search bounds
            positions[i] = np.clip(positions[i], bounds[0], bounds[1])

        # Record best fitness and solution of current generation
        best_fitness_list.append(rastrigin(global_best))
        best_solution_list.append(global_best.copy())

    # Output: Print best result per generation
    for gen, (fit, sol) in enumerate(zip(best_fitness_list, best_solution_list), 1):
        print(f"Generation {gen:3d} | Best Fitness: {fit:.6f} | x = {sol[0]:.5f}, y = {sol[1]:.5f}")

    #Save to CSV
    df = pd.DataFrame({
        "Generation": range(1, generations + 1),
        "Best_Fitness": best_fitness_list,
        "Best_X": [s[0] for s in best_solution_list],
        "Best_Y": [s[1] for s in best_solution_list]
    })
    df.to_csv("pso_fitness_data.csv", index=False)

    # Plot convergence graph
    plt.plot(best_fitness_list, color='red', label='PSO')
    plt.title("PSO Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.grid()
    plt.show()

# Run the PSO algorithm when the script is executed directly
if __name__ == "__main__":
    particle_swarm_optimization()
