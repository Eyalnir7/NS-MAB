import math
import random


class EpsilonGreedy:
    def __init__(self, n_arms, epsilon, deterministic=False):
        self.counts = [0 for _ in range(n_arms)]
        self.means = [0.0 for _ in range(n_arms)]
        self.epsilon = epsilon
        self.deterministic = deterministic
        self.t = 1

    # UCB arm selection based on max of UCB reward of each arm
    def select_arm(self):
        n_arms = len(self.counts)
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                return arm

        if self.deterministic:
            if self.t % (1/self.epsilon) != 0:
                return self.means.index(max(self.means))
            return self.means.index(min(self.means))
        if random.random() > self.epsilon:
            return self.means.index(max(self.means))
        return self.means.index(min(self.means))

    # Choose to update chosen arm and reward
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]

        # Update average/mean value/reward for chosen arm
        self.t += 1
        value = self.means[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.means[chosen_arm] = new_value

    def reset(self):
        self.counts = [0 for _ in range(len(self.counts))]
        self.means = [0.0 for _ in range(len(self.counts))]
