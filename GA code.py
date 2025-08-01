import numpy as np
import matplotlib.pyplot as plt

# Hàm mục tiêu (Sphere Function)
def fitness_function(x):
    return np.sum(x**2)

# Khởi tạo quần thể ngẫu nhiên
def initialize_population(pop_size, dimensions, bounds):
    return np.random.uniform(bounds[0], bounds[1], size=(pop_size, dimensions))

# Tính fitness cho mỗi cá thể
def evaluate_population(population):
    return np.array([fitness_function(ind) for ind in population])

# Chọn lọc: roulette wheel selection
def select_parents(population, fitness):
    inverse_fitness = 1 / (fitness + 1e-6)
    probabilities = inverse_fitness / np.sum(inverse_fitness)
    indices = np.random.choice(len(population), size=2, replace=False, p=probabilities)
    return population[indices[0]], population[indices[1]]

# Lai ghép: single-point crossover
def crossover(p1, p2):
    point = np.random.randint(1, len(p1))
    child1 = np.concatenate((p1[:point], p2[point:]))
    child2 = np.concatenate((p2[:point], p1[point:]))
    return child1, child2

# Đột biến
def mutate(ind, mutation_rate, bounds):
    for i in range(len(ind)):
        if np.random.rand() < mutation_rate:
            ind[i] = np.random.uniform(bounds[0], bounds[1])
    return ind

# GA chính
def genetic_algorithm():
    pop_size = 50
    dimensions = 2
    bounds = (-5.12, 5.12)
    generations = 100
    mutation_rate = 0.1

    population = initialize_population(pop_size, dimensions, bounds)
    best_fitness_list = []

    for gen in range(generations):
        fitness = evaluate_population(population)
        new_population = []

        # Giữ cá thể tốt nhất (elitism)
        best_idx = np.argmin(fitness)
        new_population.append(population[best_idx])
        best_fitness_list.append(fitness[best_idx])

        # Tạo quần thể mới
        while len(new_population) < pop_size:
            parent1, parent2 = select_parents(population, fitness)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate, bounds)
            child2 = mutate(child2, mutation_rate, bounds)
            new_population.extend([child1, child2])

        population = np.array(new_population[:pop_size])
        print(f"Generation {gen+1}, Best Fitness: {best_fitness_list[-1]}")

    # Vẽ đồ thị hội tụ
    plt.plot(best_fitness_list)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("GA Convergence Curve")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    genetic_algorithm()
