import math
import random
import pandas as pd
import numpy as np
import pickle

def draw(p, deterministic):
    """
    draw from ber(p) or return p if deterministic
    """
    if deterministic:
        return p
    if random.random() > p:
        return 0.0
    else:
        return 1.0
    

def run_simulation(horizon, algorithms, changes, deterministic):
    """
    :param deterministic: whether the arms are bernoulli or deterministic
    :param horizon: number of rounds
    :param algorithms: the algorithms that chooses the arms. a dictionary with the algorithm name as key and the algorithm as value
    :param changes: list of tuples. first entry in a tuple is the round of the change, with values in [1, horizon]. second entry is the new values of the arms as a list.
    returns a dictionary of list of tuples. The key is the algorithm name and the value is a list of tuples. Each tuple contains the index of the chosen arm and the reward from it.
    """
    results = {algo_name: [] for algo_name in algorithms.keys()}
    for algo in algorithms.values():
        algo.reset()

    samples = []

    phase = 0
    arm_values = changes[phase][1]
    for t in range(1, horizon+1):

        if phase+1 != len(changes) and t == changes[phase+1][0]:
            phase += 1
            arm_values = changes[phase][1]

        sample = [draw(p, deterministic) for p in arm_values]
        samples.append(sample)
        for algo_name, algo in algorithms.items():
            chosen_arm = algo.select_arm()
            reward = sample[chosen_arm]
            results[algo_name].append((chosen_arm, reward))
            algo.update(chosen_arm, reward)
    
    return results, samples
    

