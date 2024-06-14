import numpy as np


class SWUCB:
    def __init__(self, n_arms, window_size, radius_function):
        """
        Important: the count of an arm are allowed to be zero. This must be taken into consideration when implementing
        :param n_arms: amount of arms in the simulation
        :param window_size: window size for the sliding window
        :param radius_function: how the "confidence bounds" should be calculated
        """
        self.n_arms = n_arms
        self.window_size = window_size
        self.radius_function = radius_function

        # Initialize counts, values, and bounds for each arm
        self.counts = [0 for _ in range(n_arms)]
        self.values = [0.0 for _ in range(n_arms)]
        self.ucb_values = [0.0 for _ in range(n_arms)]
        self.lcb_values = [0.0 for _ in range(n_arms)]
        self.counts_log = [[] for _ in range(n_arms)]

        # Initialize a sliding window for each arm
        self.rewards_window = []

        # Time step
        self.t = 1

    def select_arm(self):
        if self.t <= self.window_size:
            for arm in range(self.n_arms):
                if self.counts[arm] == 0:
                    return arm

        # Select arm based on max of UCB reward within the sliding window
        return self.ucb_values.index(max(self.ucb_values))

    def update(self, chosen_arm, reward):
        """
        Important: the count of an arm are allowed to be zero. This must be taken into consideration when implementing
        the radius function
        :param chosen_arm: the arm the algorithm picked in the select_arm function
        :param reward: a sample from the arm from that was taken in round t
        """
        # Update counts, values, and rewards window for the chosen arm
        self.counts[chosen_arm] += 1
        self.counts_log[chosen_arm].append(self.t)
        self.rewards_window.append((chosen_arm, reward))
        flag = False
        if len(self.rewards_window) > self.window_size:
            forgot_arm, forgot_reward = self.rewards_window.pop(0)
            self.counts[forgot_arm] -= 1
            filtered_window = [t[1] for t in self.rewards_window if t[0] == forgot_arm]
            self.values[forgot_arm] = np.mean(filtered_window)
            flag = True

        if flag and forgot_arm != chosen_arm or not flag:
          filtered_window = [t[1] for t in self.rewards_window if t[0] == chosen_arm]
          self.values[chosen_arm] = np.mean(filtered_window)

        # Update UCB and LCB values for all arms
        for arm in range(self.n_arms):
            if self.counts[arm] > 0:
                radius = self.radius_function(self.t, self.counts[arm])
                self.ucb_values[arm] = self.values[arm] + radius
                self.lcb_values[arm] = self.values[arm] - radius

        # Increment time step
        self.t += 1

    def reset(self):
        # Reset counts, values, and bounds for each arm
        self.counts = [0 for _ in range(self.n_arms)]
        self.values = [0.0 for _ in range(self.n_arms)]
        self.ucb_values = [0.0 for _ in range(self.n_arms)]
        self.lcb_values = [0.0 for _ in range(self.n_arms)]

        # Reset sliding window for each arm
        self.rewards_window = []

        # Reset time step
        self.t = 1