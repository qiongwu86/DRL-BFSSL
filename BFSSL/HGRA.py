import numpy as np
import random

GHZ = 1e9  # GHz to Hz
MB = 1e6  # Megabytes to Bytes
f_max = 4e8
f_min = 5e7

class HGRA:
    def __init__(self, dim, f, v, w, a, rho, sigma, gamma, theta, epsilon, n_particles):
        self.dim = dim
        self.f = f
        self.v = v
        self.w = w
        self.a = a
        self.rho = rho
        self.sigma = sigma
        self.gamma = gamma
        self.theta = theta
        self.epsilon = epsilon
        self.n_particles = n_particles

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

            def partition_task(self):
                probabilities = self.get_probabilities()
                return probabilities / np.sum(probabilities)

            def update_weights(self, reward):
                probabilities = self.get_probabilities()
                self.reward_shift = max(self.reward_shift, abs(reward))
                shifted_reward = reward + self.reward_shift
                shifted_reward = min(shifted_reward, self.max_reward)
                estimated_reward = shifted_reward / probabilities
                self.weights = np.exp(estimated_reward)

        action_comm = np.zeros((self.dim, 3))
        action_comp = np.zeros(self.dim + 1)

        uto_exp3_comm = [UTO_EXP3(num_partitions=3, rho=self.rho) for _ in range(self.dim)]
        uto_exp3_comp = UTO_EXP3(num_partitions=4, rho=self.rho)

        for _ in range(iterations):
            for i, uto_exp3_comm_i in enumerate(uto_exp3_comm):
                comm_i = uto_exp3_comm_i.partition_task()
                action_comm[i] = comm_i

            action_comp = uto_exp3_comp.partition_task()
            action_comp = [int(x*(f_max-f_min)+f_min) for x in action_comp]  # 限制范围在 [5e7, 4e8] 内

        return action_comp