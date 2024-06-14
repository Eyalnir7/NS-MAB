from algorithms.DUCB import DUCB
from simulation import run_simulation
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from algorithms.UCB1 import UCB1
from algorithms.SWUCB import SWUCB
from algorithms.UCB import UCB
import random

random.seed(639)
matplotlib.use('TkAgg')


def calculate_cum_regret(results, samples, changes, horizon):
    """
    :param results: dictionary of list of tuples. The key is the algorithm name and the value is a list of tuples. Each tuple contains the index of the chosen arm and the reward from it.
    """
    cum_regret = {algo_name: np.zeros(horizon) for algo_name in results.keys()}
    phase = 0
    arm_values = changes[phase][1]
    max_arm = arm_values.index(max(arm_values))
    for t in range(horizon):

        if phase + 1 != len(changes) and t == changes[phase + 1][0] - 1:
            phase += 1
            arm_values = changes[phase][1]
            max_arm = arm_values.index(max(arm_values))

        for algo_name, algo_results in results.items():
            cum_regret[algo_name][t] = samples[t][max_arm] - arm_values[algo_results[t][0]] + cum_regret[algo_name][
                t - 1] if t != 0 else samples[t][max_arm] - arm_values[algo_results[t][0]]
    return cum_regret


def check_results_ducb(results, samples, round, gamma, zeta):
    """
    :param results: results from the simulation
    :param samples: samples from the simulation
    :param round: the round to check the choise of ducb in. round in [1, horizon-2]
    (think of it like the first round is 0). it will look at the samples at all previous rounds and check whether
    the ducb choice was correct.
    """
    ducb_choices = results["DUCB"]
    dsums = np.zeros(len(samples[0]))
    dcounts = np.zeros(len(samples[0]))
    if round == 0:
        print("No previous rounds to check")
        return True
    for t in range(round):
        chosen_arm = ducb_choices[t][0]
        reward = ducb_choices[t][1]
        dcounts[chosen_arm] += 1 * (gamma ** (round - t - 1))
        dsums[chosen_arm] += reward * (gamma ** (round - t - 1))

    n_t = sum(dcounts)
    c = 2 * np.sqrt((zeta * np.log(n_t)) / dcounts)
    ucb_values = dsums / dcounts + c
    print(ucb_values)
    chosen_arm_real = np.argmax(ucb_values)
    chosen_arm_ducb = ducb_choices[round][0]
    print(f"chosen_arm_real = {chosen_arm_real}, chosen_arm_ducb = {chosen_arm_ducb}")
    return chosen_arm_real == chosen_arm_ducb


def test1():
    N = 10
    horizon = 10000
    gamma = 1 - 0.25 * np.sqrt((1 / horizon))
    algo = DUCB(0.6, 3, gamma)
    tau = 4 * np.sqrt(horizon * np.log(horizon))
    sw_radius_function = lambda t, c: np.sqrt((0.6 * np.log10(min(t, tau))) / (c))
    sw_ucb = SWUCB(3, tau, sw_radius_function)
    usb_radius_function = lambda t, c: np.sqrt((0.5 * np.log10(t)) / (c))
    ucb = UCB(3, usb_radius_function)
    algorithms = {"DUCB": algo, "SWUCB": sw_ucb, "UCB": ucb}
    changes = [(1, [0.5, 0.3, 0.4]), (3001, [0.5, 0.3, 0.9]), (5001, [0.5, 0.3, 0.4])]
    cum_regret = {algo_name: np.zeros(horizon) for algo_name in algorithms.keys()}
    for i in range(N):
        results, samples = run_simulation(horizon, algorithms, changes, False)
        # print(check_results_ducb(results, samples, 4, gamma, 0.6))
        current_cum_regret = calculate_cum_regret(results, samples, changes, horizon)
        for algo_name in algorithms.keys():
            cum_regret[algo_name] += 1 / N * current_cum_regret[algo_name]

    plt.plot(cum_regret["DUCB"], label="DUCB")
    plt.plot(cum_regret["SWUCB"], label="SWUCB")
    plt.plot(cum_regret["UCB"], label="UCB")
    plt.legend()
    plt.show()


def test2():
    N = 1
    horizon = 10000
    gamma = 1 - 0.25 * np.sqrt((1 / horizon))
    algo = DUCB(0.6, 2, gamma)
    tau = 4 * np.sqrt(horizon * np.log(horizon))
    sw_radius_function = lambda t, c: np.sqrt((0.6 * np.log10(min(t, tau))) / (c))
    sw_ucb = SWUCB(2, tau, sw_radius_function)
    usb_radius_function = lambda t, c: np.sqrt((0.5 * np.log10(t)) / (c))
    ucb = UCB(2, usb_radius_function)
    algorithms = {"DUCB": algo, "SWUCB": sw_ucb, "UCB": ucb}
    changes = [(1, [0, 0]), (3001, [0.5, 1])]
    cum_regret = {algo_name: np.zeros(horizon) for algo_name in algorithms.keys()}
    for i in range(N):
        results, samples = run_simulation(horizon, algorithms, changes, True)
        # print(check_results_ducb(results, samples, 4, gamma, 0.6))
        current_cum_regret = calculate_cum_regret(results, samples, changes, horizon)
        for algo_name in algorithms.keys():
            cum_regret[algo_name] += 1 / N * current_cum_regret[algo_name]

    plt.plot(cum_regret["DUCB"], label="DUCB")
    plt.plot(cum_regret["SWUCB"], label="SWUCB")
    plt.plot(cum_regret["UCB"], label="UCB")
    plt.legend()
    plt.show()

test1()