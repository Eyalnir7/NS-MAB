import math
import numpy as np


class DUCB:
    def __init__(self, alpha, n_arms, discount_factor):
        self.counts = [0 for _ in range(n_arms)]
        self.values = [0.0 for _ in range(n_arms)]
        self.dcounts = np.zeros(n_arms)
        self.dsums = np.zeros(n_arms)
        self.alpha = alpha
        self.ucb_values = [0.0 for arm in range(n_arms)]
        self.lcb_values = [0.0 for arm in range(n_arms)]
        self.t = 1
        self.discount_factor = discount_factor
        return

    # UCB arm selection based on max of UCB reward of each arm
    def select_arm(self):
        n_arms = len(self.counts)
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                return arm

        return self.ucb_values.index(max(self.ucb_values))

    # Choose to update chosen arm and reward
    def update(self, chosen_arm, reward):
        self.dcounts = self.dcounts * self.discount_factor
        self.dcounts[chosen_arm] += 1
        self.dsums = self.dsums * self.discount_factor
        self.dsums[chosen_arm] += reward

        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value

        for arm in range(len(self.ucb_values)):
            if self.counts[arm] != 0:
                radius = 2*math.sqrt((self.alpha * np.log10(np.sum(self.dcounts))) / self.dcounts[arm])
                self.ucb_values[arm] = self.dsums[arm]/self.dcounts[arm] + radius
                self.lcb_values[arm] = self.dsums[arm]/self.dcounts[arm] - radius
        self.t += 1

    def reset(self):
        num_arms = len(self.counts)
        self.counts = [0 for _ in range(num_arms)]
        self.values = [0.0 for _ in range(num_arms)]
        self.ucb_values = [0.0 for arm in range(num_arms)]
        self.lcb_values = [0.0 for arm in range(num_arms)]
        self.t = 1
        self.dcounts = np.zeros(num_arms)
        self.dsums = np.zeros(num_arms)