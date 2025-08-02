import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  #Import pandas to save CSV
from init_population import initialize_population  # Custom function to initialize the population

# Definition of the Rastrigin function
def rastrigin(x):
    return 20 + x[0]**2 + x[1]**2 - 10 * (np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))
    # The function has a global minimum at (0,0) with value 0

# Evaluate each individual in the population using the Rastrigin function
def evaluate_population(population):
    return np.array([rastrigin(ind) for ind in population])

# Select two parents based on fitness-proportionate selection
def select_parents(pop, fitness):
    inv = 1 / (fitness + 1e-6)   # Invert the fitness values
    prob = inv / np.sum(inv)    # Normalize to get selection probabilities
    ids = np.random.choice(len(pop), 2, replace=False, p=prob)  # Select 2 unique parents based on probabilities
    return pop[ids[0]], pop[ids[1]]  # Return the selected parent individuals

# Perform one-point crossover between two parents to produce two children
def crossover(p1, p2):
    point = np.random.randint(1, len(p1))  # Random crossover point
    # Combine parts of parents to create children
    return np.concatenate((p1[:point], p2[point:])), np.concatenate((p2[:point], p1[point:]))

# Mutate an individual by replacing gene values with a new random value with some probability
def mutate(ind, rate, bounds):
    for i in range(len(ind)):
        if np.random.rand() < rate:  # Mutation occurs based on the mutation rate
            ind[i] = np.random.uniform(bounds[0], bounds[1])  # Replace gene with random value within bounds
    return ind

# Main function implementing the Genetic Algorithm
def genetic_algorithm():
    pop_size = 50                 # Number of individuals in the population
    dimensions = 2                # Problem dimensionality
    bounds = (-5.12, 5.12)        # Search space bounds
    generations = 100             # Number of iterations/generations
    mutation_rate = 0.1           # Probability of mutation per gene

    # Initialize population using a helper function
    population = initialize_population(pop_size, dimensions, bounds)
    best_fitness_list = []        # Track best fitness per generation
    best_solution_list = []       # Track best solution per generation

    # Main GA loop
    for gen in range(generations):
        fitness = evaluate_population(population)  # Evaluate current population
        new_population = []       # Create a new population

        best_idx = np.argmin(fitness)  # Find index of the best individual (lowest fitness)
        best_solution = population[best_idx]  # Get best solution
        best_fitness_list.append(fitness[best_idx])  # Record best fitness
        best_solution_list.append(best_solution.copy())  # Save best solution
        new_population.append(best_solution)  # Apply elitism: carry best to next generation

        # Generate rest of the new population
        while len(new_population) < pop_size:
            p1, p2 = select_parents(population, fitness)  # Select parents
            c1, c2 = crossover(p1, p2)                     # Crossover to generate children
            c1 = mutate(c1, mutation_rate, bounds)         # Mutate child 1
            c2 = mutate(c2, mutation_rate, bounds)         # Mutate child 2
            new_population.extend([c1, c2])                # Add children to the new population

        population = np.array(new_population[:pop_size])  # Replace old population with new one (trimming if needed)

    # Print and log best solution per generation
    for gen, (fit, sol) in enumerate(zip(best_fitness_list, best_solution_list), 1):
        print(f"Generation {gen:3d} | Best Fitness: {fit:.6f} | x = {sol[0]:.5f}, y = {sol[1]:.5f}")

    #Save fitness data to CSV
    df = pd.DataFrame({
        "Generation": range(1, generations + 1),
        "Best_Fitness": best_fitness_list,
        "Best_X": [s[0] for s in best_solution_list],
        "Best_Y": [s[1] for s in best_solution_list]
    })
    df.to_csv("ga_fitness_data.csv", index=False)

    # Plot convergence graph
    plt.plot(best_fitness_list, label='GA', color='blue')
    plt.title("GA Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.grid()
    plt.show()

# Run the GA when the script is executed
if __name__ == "__main__":
    genetic_algorithm()
