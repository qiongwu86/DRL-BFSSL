import numpy as np
import random

GHZ = 1e9  # GHz to Hz
MB = 1e6  # Megabytes to Bytes
f_max = 4e8
f_min = 5e7

class HGGA:
    def __init__(self, dim, population_size, mutation_rate, n_generations, rho):
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.n_generations = n_generations
        self.rho = rho

    def init_population(self):
        return np.array([self.initPos(self.dim) for _ in range(self.population_size)])

    def initPos(self, n):
        vector = [random.uniform(0, 1) for _ in range(n)]
        return vector

    def fitness(self, individual):
        # Define a fitness function based on your specific problem
        # Example: Here we just sum the individual's elements
        return np.sum(individual)  # Replace with your fitness calculation

    def select(self, population):
        fitness_values = np.array([self.fitness(ind) for ind in population])
        probabilities = fitness_values / np.sum(fitness_values)
        selected_indices = np.random.choice(np.arange(self.population_size), size=self.population_size, p=probabilities)
        return population[selected_indices]

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, self.dim - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2

    def mutate(self, individual):
        for i in range(self.dim):
            if random.random() < self.mutation_rate:
                individual[i] = random.uniform(0, 1)
        return individual
    def run(self):
        population = self.init_population()

        for generation in range(self.n_generations):
            selected_population = self.select(population)
            next_generation = []

            for i in range(0, self.population_size, 2):
                parent1 = selected_population[i]
                parent2 = selected_population[min(i + 1, self.population_size - 1)]
                child1, child2 = self.crossover(parent1, parent2)
                next_generation.append(self.mutate(child1))
                next_generation.append(self.mutate(child2))

            population = np.array(next_generation)

        best_individual = max(population, key=self.fitness)
        action_comp = [int(x * (f_max - f_min) + f_min) for x in best_individual]  # 限制范围在 [5e7, 4e8] 内
        return action_comp
