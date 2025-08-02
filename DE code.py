import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  #Add pandas for CSV export
from init_population import initialize_population  # Custom function to create initial population

# Rastrigin function definition (a non-convex benchmark function with many local minima)
def rastrigin(x):
    return 20 + x[0]**2 + x[1]**2 - 10 * (np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))
    # Global minimum is at (0, 0) with function value 0

# Main function implementing Differential Evolution algorithm
def differential_evolution():
    pop_size = 50              # Number of individuals in the population
    dimensions = 2             # Number of decision variables (2D optimization)
    bounds = (-5.12, 5.12)     # Lower and upper bounds for each dimension
    generations = 100          # Number of iterations/generations
    F = 0.5                    # Differential weight (mutation factor)
    CR = 0.9                   # Crossover probability

    # Initialize population randomly within bounds
    population = initialize_population(pop_size, dimensions, bounds)

    # Evaluate fitness of the initial population
    fitness = np.array([rastrigin(ind) for ind in population])

    # Lists to store best fitness and solution for each generation
    best_fitness_list = []
    best_solution_list = []

    # Main evolution loop
    for _ in range(generations):
        new_population = []  # Container for the next generation

        for i in range(pop_size):
            # Select three other distinct individuals a, b, c
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]

            # Create mutant vector using differential mutation
            mutant = a + F * (b - c)

            # Clip mutant to be within the search bounds
            mutant = np.clip(mutant, bounds[0], bounds[1])

            # Perform binomial crossover between target and mutant
            cross_points = np.random.rand(dimensions) < CR  # Boolean mask for crossover
            if not np.any(cross_points):                   # Ensure at least one gene is crossed
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, population[i])  # Trial vector

            # Evaluate trial individual
            trial_fitness = rastrigin(trial)

            # Selection: if trial is better, replace target
            if trial_fitness < fitness[i]:
                new_population.append(trial)
                fitness[i] = trial_fitness
            else:
                new_population.append(population[i])

        # Update the population for next generation
        population = np.array(new_population)

        # Record best solution in the current generation
        best_idx = np.argmin(fitness)
        best_fitness_list.append(fitness[best_idx])
        best_solution_list.append(population[best_idx].copy())

    # Print best result from each generation
    for gen, (fit, sol) in enumerate(zip(best_fitness_list, best_solution_list), 1):
        print(f"Generation {gen:3d} | Best Fitness: {fit:.6f} | x = {sol[0]:.5f}, y = {sol[1]:.5f}")

    #Save raw fitness data to CSV
    df = pd.DataFrame({
        "Generation": range(1, generations + 1),
        "Best_Fitness": best_fitness_list,
        "Best_X": [s[0] for s in best_solution_list],
        "Best_Y": [s[1] for s in best_solution_list]
    })
    df.to_csv("de_fitness_data.csv", index=False)

    # Plot convergence graph
    plt.plot(best_fitness_list, color='green', label='DE')
    plt.title("DE Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.grid()
    plt.show()

# Run the DE algorithm if script is executed directly
if __name__ == "__main__":
    differential_evolution()
