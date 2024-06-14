import math


class UCB:
    def __init__(self, n_arms, radius_function):
        self.counts = [0 for _ in range(n_arms)]
        self.values = [0.0 for _ in range(n_arms)]
        self.ucb_values = [0.0 for arm in range(n_arms)]
        self.lcb_values = [0.0 for arm in range(n_arms)]
        self.t = 1
        self.f = radius_function
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
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]

        # Update average/mean value/reward for chosen arm
        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
        for arm in range(len(self.ucb_values)):
            if self.counts[arm] != 0:
                radius = self.f(self.t, self.counts[arm])
                self.ucb_values[arm] = self.values[arm] + radius
                self.lcb_values[arm] = self.values[arm] - radius
        self.t += 1

    def reset(self):
        self.counts = [0 for _ in range(len(self.counts))]
        self.values = [0.0 for _ in range(len(self.counts))]
        self.ucb_values = [0.0 for arm in range(len(self.counts))]
        self.lcb_values = [0.0 for arm in range(len(self.counts))]
        self.t = 1