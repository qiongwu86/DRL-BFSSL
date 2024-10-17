import numpy as np
import random

class GIBBS:
    def __init__(self, dim, f, v, w, a, q, rho, sigma, gamma, theta, epsilon, n_particles):
        self.dim = dim
        self.f = f
        self.v = v
        self.w = w
        self.a = a
        self.q = q
        self.rho = rho
        self.c = 1
        self.w1 = 1 / 1e2
        self.w2 = 1 / 1e5
        self.sigma = sigma
        self.gamma = gamma
        self.theta = theta
        self.epsilon = epsilon
        self.Task_require = [30, 30, 30, 30, 30, 30, 30, 30]

        # CPU frequency range
        self.cpu_freq_min = 5e7
        self.cpu_freq_max = 4e8

    def compute_cost_fct(self, cores, cpu_usage):  # core = 10
        return cores * (cpu_usage / 400 / self.cpu_freq_max / cores) ** 3 * 1000

    def compute_cost(self, action_comp, action_comm):
        cost = 0
        for xi, ai, qi, wi in zip(action_comp, self.a, self.q, self.w):
            cost += (ai * qi / 1e6 - qi * self.f * xi / wi / 1e6) / 1e3
        cost_comp = self.compute_cost_fct(10, self.f * sum(action_comp)) * self.w2
        cost_comm = 0
        for task_comm in action_comm:
            for ai in self.a:
                cost_comm += task_comm[2] * ai * self.c * self.w1 * 500

        cost = cost
        return self.v * cost + (cost_comp + cost_comm)

    def initPos(self, n):
        vector = [random.uniform(0, 1) for _ in range(n)]
        total_sum = sum(vector)
        normalized_vector = [v / total_sum for v in vector]
        return normalized_vector

    def run(self, iterations):
        class UTO_EXP3:
            def __init__(self, num_partitions, rho):
                self.num_partitions = num_partitions
                self.rho = rho
                self.weights = np.random.random(num_partitions)
                self.rewards = np.zeros(num_partitions)
                self.reward_shift = 0
                self.max_reward = 1e6

            def get_probabilities(self):
                total_weight = np.sum(self.weights)
                probabilities = (1 - self.rho) * (self.weights / total_weight)
                probabilities += self.rho / self.num_partitions
                return probabilities

            def select_partition(self):
                probabilities = self.get_probabilities()
                return probabilities

            def update_weights(self, partition, reward):
                probabilities = self.get_probabilities()
                self.reward_shift = max(self.reward_shift, abs(reward))
                shifted_reward = reward + self.reward_shift
                shifted_reward = min(shifted_reward, self.max_reward)
                estimated_reward = shifted_reward / probabilities
                self.rewards = estimated_reward
                self.weights = np.exp(self.rewards)

            def partition_task(self, task):
                partition = self.select_partition()
                return partition / np.sum(partition)

        uto_exp3_comm = [UTO_EXP3(num_partitions=3, rho=0.2) for _ in range(self.dim)]
        uto_exp3_comp = UTO_EXP3(num_partitions=4, rho=0.2)
        action_comm = np.zeros((self.dim, 3))
        action_comp = np.zeros(self.dim)

        for _ in range(iterations):
            for i, uto_exp3_comm_i in enumerate(uto_exp3_comm):
                comm_i = uto_exp3_comm_i.partition_task(uto_exp3_comm_i)
                action_comm[i] = comm_i

            action_comp = uto_exp3_comp.partition_task(uto_exp3_comp)
        # Map action_comp values to CPU frequency range [5e7, 4e8]
        final_cpu_frequencies = [min(max(self.cpu_freq_min, self.cpu_freq_max * xi), self.cpu_freq_max) for xi in action_comp]
        # print(f"Final allocated CPU frequencies: {final_cpu_frequencies}")

        return final_cpu_frequencies