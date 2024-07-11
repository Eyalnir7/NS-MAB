from matplotlib import gridspec
from tqdm import tqdm
from algorithms.DUCB import DUCB
from algorithms.epsilon_greedy import EpsilonGreedy
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


def calculate_cumulant_armpicks(results, horizon, num_arms):
    """
    calculate the cumulant armpicks for each algorithm and each arm given data about the simulation. By cumulant armpicks
    I mean the number of times an arm was picked up to round t.
    :param results: dictionary of list of tuples. The key is the algorithm name and the value is a list of tuples.
    Each tuple contains the index of the chosen arm and the reward from it.
    :param horizon: number of rounds
    :param num_arms: number of arms
    :return: a dictionary of list of lists of integers. The key is the algorithm name and the value is a list containing
    num_arms lists. each list contains the number of times an arm was picked up to round t.
    """
    cumulant_armpicks = {algo_name: [np.zeros(horizon) for arm in range(num_arms)] for algo_name in results.keys()}
    for t in range(horizon):
        for algo_name, algo_results in results.items():
            chosen_arm = algo_results[t][0]
            for arm in range(num_arms):
                if t > 0:
                    cumulant_armpicks[algo_name][arm][t] = cumulant_armpicks[algo_name][arm][t - 1]
                if arm == chosen_arm:
                    cumulant_armpicks[algo_name][chosen_arm][t] += 1
    return cumulant_armpicks


def calculate_cumulant_regret(results, samples, changes, horizon):
    """
    calculate the cumulant regret for each algorithm given data about the simulation
    :param results: dictionary of list of tuples. The key is the algorithm name and the value is a list of tuples.
    Each tuple contains the index of the chosen arm and the reward from it.
    :param samples: list of samples from the arms from the simulation
    :param changes: list of tuples. first entry in a tuple is the round of the change, with values in [1, horizon].
    second entry is the new values of the arms as a list.
    :param horizon: number of rounds
    """
    cumulant_regret = {algo_name: np.zeros(horizon) for algo_name in results.keys()}
    phase = 0
    arm_values = changes[phase][1]
    max_arm = arm_values.index(max(arm_values))
    for t in range(horizon):

        if phase + 1 != len(changes) and t == changes[phase + 1][0] - 1:
            phase += 1
            arm_values = changes[phase][1]
            max_arm = arm_values.index(max(arm_values))

        for algo_name, algo_results in results.items():
            cumulant_regret[algo_name][t] = arm_values[max_arm] - arm_values[algo_results[t][0]] + \
                                            cumulant_regret[algo_name][
                                                t - 1] if t != 0 else arm_values[max_arm] - arm_values[
                algo_results[t][0]]
    return cumulant_regret


def check_results_ducb(results, samples, round, gamma, zeta):
    """
    check whether the ducb choice was correct in a given round
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
    c = 0
    if round >= len(samples[0]):
        c = 2 * np.sqrt((zeta * np.log10(n_t)) / dcounts)
    ucb_values = dsums / dcounts + c
    print(ucb_values)
    chosen_arm_real = np.argmax(ucb_values)
    chosen_arm_ducb = ducb_choices[round][0]
    print(f"chosen_arm_real = {chosen_arm_real}, chosen_arm_ducb = {chosen_arm_ducb}")
    return chosen_arm_real == chosen_arm_ducb


def test1():
    N = 1
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
    cumulant_regret = {algo_name: np.zeros(horizon) for algo_name in algorithms.keys()}
    for i in range(N):
        results, samples = run_simulation(horizon, algorithms, changes, True)
        # print(check_results_ducb(results, samples, 4, gamma, 0.6))
        current_cum_regret = calculate_cumulant_regret(results, samples, changes, horizon)
        for algo_name in algorithms.keys():
            cumulant_regret[algo_name] += 1 / N * current_cum_regret[algo_name]

    plt.plot(cumulant_regret["DUCB"], label="DUCB")
    plt.plot(cumulant_regret["SWUCB"], label="SWUCB")
    plt.plot(cumulant_regret["UCB"], label="UCB")
    plt.legend()
    plt.show()


def test2():
    N = 1
    horizon = 10000
    gamma = 1 - 0.25 * np.sqrt((1 / horizon))
    algo = DUCB(1, 2, gamma)
    tau = 4 * np.sqrt(horizon * np.log(horizon))
    sw_radius_function = lambda t, c: np.sqrt((0.6 * np.log10(min(t, tau))) / (c))
    sw_ucb = SWUCB(2, tau, sw_radius_function)
    usb_radius_function = lambda t, c: np.sqrt((0.5 * np.log10(t)) / (c))
    ucb = UCB(2, usb_radius_function)
    algorithms = {"DUCB": algo, "UCB": ucb}
    changes = [(1, [1, 0]), (3001, [0, 1]), (5001, [1, 0])]
    cumulant_regret = {algo_name: np.zeros(horizon) for algo_name in algorithms.keys()}
    for i in range(N):
        results, samples = run_simulation(horizon, algorithms, changes, True)
        armpicks = calculate_cumulant_armpicks(results, horizon, 2)
        plt.plot(armpicks["DUCB"][0], label="DUCB arm 0")
        plt.plot(armpicks["DUCB"][1], label="DUCB arm 1")
        # plt.plot(cumulant_regret["SWUCB"], label="SWUCB")
        plt.plot(armpicks["UCB"][1], label="UCB arm 1")
        plt.plot(armpicks["UCB"][0], label="UCB arm 0")
        plt.legend()
        plt.show()

        current_cum_regret = calculate_cumulant_regret(results, samples, changes, horizon)
        for algo_name in algorithms.keys():
            cumulant_regret[algo_name] += 1 / N * current_cum_regret[algo_name]

    plt.plot(cumulant_regret["DUCB"], label="DUCB")
    # plt.plot(cumulant_regret["SWUCB"], label="SWUCB")
    plt.plot(cumulant_regret["UCB"], label="UCB")
    plt.legend()
    plt.show()


def test3():
    N = 1
    horizon = 10000
    gamma = 1 - 0.25 * np.sqrt((1 / horizon))
    algo = DUCB(0.6, 2, gamma)
    tau = np.floor(4 * np.sqrt(horizon * np.log(horizon))) * 2
    sw_radius_function = lambda t, c: np.sqrt((0.6 * np.log10(min(t, tau))) / (c))
    sw_ucb = SWUCB(2, tau, sw_radius_function)
    usb_radius_function = lambda t, c: np.sqrt((0.5 * np.log10(t)) / (c))
    ucb = UCB(2, usb_radius_function)
    delta = np.sqrt((0.85 * tau - 2) * np.log(tau) / (2 * (0.15 * tau - 1.5) ** 2))
    c = np.floor(0.3 * tau) if 0.3 * tau % 2 == 1 else np.floor(0.3 * tau)
    algorithms = {"DUCB": algo, "SWUCB": sw_ucb, "UCB": ucb}
    changes = [(1, [0, 0])]
    i = 0
    while True:
        if i * tau + c + i + 1 > horizon:
            break
        changes.append((i * tau + c + i, [delta, 1]))
        if (i + 1) * tau + i + 1 > horizon:
            break
        changes.append(((i + 1) * tau + i + 1, [delta, 0]))
        i += 1
    print(len(changes))
    cumulant_regret = {algo_name: np.zeros(horizon) for algo_name in algorithms.keys()}
    for i in range(N):
        results, samples = run_simulation(horizon, algorithms, changes, True)
        # print(check_results_ducb(results, samples, 4, gamma, 0.6))
        armpicks = calculate_cumulant_armpicks(results, horizon, 2)
        plt.plot(armpicks["DUCB"][0], label="DUCB arm 0")
        plt.plot(armpicks["DUCB"][1], label="DUCB arm 1")
        # plt.plot(cumulant_regret["SWUCB"], label="SWUCB")
        # plt.plot(armpicks["UCB"][1], label="UCB arm 1")
        # plt.plot(armpicks["UCB"][0], label="UCB arm 0")
        plt.plot(armpicks["SWUCB"][1], label="SWUCB arm 1")
        plt.plot(armpicks["SWUCB"][0], label="SWUCB arm 0")
        for x in [i[0] for i in changes]:
            plt.axvline(x=x, color='red', linestyle='--', linewidth=0.8)
        plt.legend()
        plt.show()
        current_cum_regret = calculate_cumulant_regret(results, samples, changes, horizon)
        for algo_name in algorithms.keys():
            cumulant_regret[algo_name] += 1 / N * current_cum_regret[algo_name]

    plt.plot(cumulant_regret["DUCB"], label="DUCB")
    plt.plot(cumulant_regret["SWUCB"], label="SWUCB")
    # plt.plot(cumulant_regret["UCB"], label="UCB")
    for x in [i[0] for i in changes]:
        plt.axvline(x=x, color='red', linestyle='--', linewidth=0.8)
    print(cumulant_regret["SWUCB"] == cumulant_regret["UCB"])
    plt.legend()
    plt.show()


def test4():
    N = 1
    horizon = 10000
    gamma = 1 - 0.25 * np.sqrt((1 / horizon))
    algo = DUCB(0.6, 2, gamma)
    tau = np.floor(4 * np.sqrt(horizon * np.log(horizon))) * 2
    sw_radius_function = lambda t, c: np.sqrt((0.6 * np.log10(min(t, tau))) / (c))
    sw_ucb = SWUCB(2, tau, sw_radius_function)
    usb_radius_function = lambda t, c: np.sqrt((0.5 * np.log10(t)) / (c))
    ucb = UCB(2, usb_radius_function)
    delta = np.sqrt((0.85 * tau - 2) * np.log(tau) / (2 * (0.15 * tau - 1.5) ** 2))
    c = np.floor(0.3 * tau) if 0.3 * tau % 2 == 1 else np.floor(0.3 * tau) + 1
    algorithms = {"DUCB": algo, "SWUCB": sw_ucb, "UCB": ucb}
    changes = [(i * horizon // 10 + 1, [0, 1]) if i % 4 == 0 else (i * horizon // 10 + 1, [1, 0]) for i in range(10)]
    cumulant_regret = {algo_name: np.zeros(horizon) for algo_name in algorithms.keys()}
    for i in range(N):
        results, samples = run_simulation(horizon, algorithms, changes, True)
        # print(check_results_ducb(results, samples, 4, gamma, 0.6))
        current_cum_regret = calculate_cumulant_regret(results, samples, changes, horizon)

        for algo_name in algorithms.keys():
            cumulant_regret[algo_name] += 1 / N * current_cum_regret[algo_name]

    plt.plot(cumulant_regret["DUCB"], label="DUCB")
    plt.plot(cumulant_regret["SWUCB"], label="SWUCB")
    plt.plot(cumulant_regret["UCB"], label="UCB")
    print(cumulant_regret["SWUCB"] == cumulant_regret["UCB"])
    plt.legend()
    plt.show()


def test5():
    N = 1
    horizon = 10000
    gamma = 1 - 0.25 * np.sqrt((1 / horizon))
    algo = DUCB(2, 2, gamma)
    usb_radius_function = lambda t, c: np.sqrt((0.5 * np.log(t)) / (c))
    ucb = UCB(2, usb_radius_function)
    delta = np.sqrt((0.85 * tau - 2) * np.log(tau) / (2 * (0.15 * tau - 1.5) ** 2))
    c = np.floor(0.3 * tau) if 0.3 * tau % 2 == 1 else np.floor(0.3 * tau) + 1
    algorithms = {"DUCB": algo, "SWUCB": sw_ucb, "UCB": ucb}
    changes = [(1, (0, 0)), (2001, (delta, 1))]
    cumulant_regret = {algo_name: np.zeros(horizon) for algo_name in algorithms.keys()}
    for i in range(N):
        results, samples = run_simulation(horizon, algorithms, changes, True)
        armpicks = calculate_cumulant_armpicks(results, horizon, 2)
        plt.plot(armpicks["DUCB"][0], label="DUCB arm 0")
        plt.plot(armpicks["DUCB"][1], label="DUCB arm 1")
        plt.plot(armpicks["SWUCB"][0], label="SWUCB arm 0")
        plt.plot(armpicks["SWUCB"][1], label="SWUCB arm 1")
        plt.plot(armpicks["UCB"][1], label="UCB arm 1")
        plt.plot(armpicks["UCB"][0], label="UCB arm 0")
        plt.legend()
        plt.show()
        # print(check_results_ducb(results, samples, 4, gamma, 0.6))
        current_cum_regret = calculate_cumulant_regret(results, samples, changes, horizon)
        for algo_name in algorithms.keys():
            cumulant_regret[algo_name] += 1 / N * current_cum_regret[algo_name]

    plt.plot(cumulant_regret["DUCB"], label="DUCB")
    plt.plot(cumulant_regret["SWUCB"], label="SWUCB")

    plt.plot(cumulant_regret["UCB"], label="UCB")
    plt.legend()
    plt.show()


def test_for_deterministic_probability_thm():
    N = 1000
    horizon = 100
    usb_radius_function = lambda t, c: np.sqrt((0.5 * np.log(t)) / (c))
    ucb = UCB(2, usb_radius_function)
    algorithms = {"UCB": ucb}
    changes = [(1, [0.1, 0.1]), (51, [0.5, 1])]
    cumulant_regret = {algo_name: np.zeros(horizon) for algo_name in algorithms.keys()}
    for i in range(N):
        results, samples = run_simulation(horizon, algorithms, changes, False)
        current_cum_regret = calculate_cumulant_regret(results, samples, changes, horizon)
        for algo_name in algorithms.keys():
            cumulant_regret[algo_name] += 1 / N * current_cum_regret[algo_name]

    plt.plot(cumulant_regret["UCB"], label="UCB")
    plt.legend()
    plt.show()


def test6():
    N = 1
    horizon = 10000
    gamma = 1 - 0.25 * np.sqrt((1 / horizon))
    algo = DUCB(0.6, 2, gamma)
    tau = np.floor(4 * np.sqrt(horizon * np.log(horizon))) * 2
    sw_radius_function = lambda t, c: np.sqrt((0.6 * np.log10(min(t, tau))) / (c))
    sw_ucb = SWUCB(2, tau, sw_radius_function)
    usb_radius_function = lambda t, c: np.sqrt((0.5 * np.log10(t)) / (c))
    ucb = UCB(2, usb_radius_function)
    delta = np.sqrt((0.85 * tau - 2) * np.log(tau) / (2 * (0.15 * tau - 1.5) ** 2))
    c = np.floor(0.3 * tau) if 0.3 * tau % 2 == 1 else np.floor(0.3 * tau)
    algorithms = {"DUCB": algo, "SWUCB": sw_ucb, "UCB": ucb}
    changes = [(1, [0, 0.5])]
    # i = 0
    # while True:
    #     if i * tau + c + i + 1 > horizon:
    #         break
    #     changes.append((i * tau + c + i, [delta, 1]))
    #     if (i + 1) * tau + i + 1 > horizon:
    #         break
    #     changes.append(((i + 1) * tau + i + 1, [delta, 0]))
    #     i += 1
    print(len(changes))
    cumulant_regret = {algo_name: np.zeros(horizon) for algo_name in algorithms.keys()}
    for i in range(N):
        results, samples = run_simulation(horizon, algorithms, changes, True)
        # print(check_results_ducb(results, samples, 4, gamma, 0.6))
        armpicks = calculate_cumulant_armpicks(results, horizon, 2)
        plt.plot(armpicks["DUCB"][0], label="DUCB arm 0")
        plt.plot(armpicks["DUCB"][1], label="DUCB arm 1")
        plt.plot(armpicks["UCB"][1], label="UCB arm 1")
        plt.plot(armpicks["UCB"][0], label="UCB arm 0")
        plt.plot(armpicks["SWUCB"][1], label="SWUCB arm 1")
        plt.plot(armpicks["SWUCB"][0], label="SWUCB arm 0")
        for x in [i[0] for i in changes]:
            plt.axvline(x=x, color='red', linestyle='--', linewidth=0.8)
        plt.legend()
        plt.show()
        current_cum_regret = calculate_cumulant_regret(results, samples, changes, horizon)
        for algo_name in algorithms.keys():
            cumulant_regret[algo_name] += 1 / N * current_cum_regret[algo_name]

    plt.plot(cumulant_regret["DUCB"], label="DUCB")
    plt.plot(cumulant_regret["SWUCB"], label="SWUCB")
    plt.plot(cumulant_regret["UCB"], label="UCB")
    for x in [i[0] for i in changes]:
        plt.axvline(x=x, color='red', linestyle='--', linewidth=0.8)
    # print(cumulant_regret["SWUCB"] == cumulant_regret["UCB"])
    plt.legend()
    plt.show()


def eps_greedy_test():
    horizon = 1000
    epsilon = 0.01
    algo = EpsilonGreedy(2, epsilon, True)
    algorithms = {"epsilon-Greedy": algo}
    changes = [(1, [0, 1]), (horizon // 2, [1, 0])]

    results, samples = run_simulation(horizon, algorithms, changes, True)
    current_cum_regret = calculate_cumulant_regret(results, samples, changes, horizon)

    fig = plt.figure(figsize=(15, 7))

    # Define GridSpec
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[:, 0])
    ax1.plot(current_cum_regret["epsilon-Greedy"], label="non-stationary deterministic epsilon-Greedy", color="orange")
    ax1.set_xlabel("Rounds", fontsize=15)
    ax1.set_ylabel("Cumulative Regret", fontsize=15)
    ax1.set_title("Cumulative Regret of epsilon-Greedy", fontsize=18)
    changes = [(1, [0, 1])]

    results, samples = run_simulation(horizon, algorithms, changes, True)
    current_cum_regret = calculate_cumulant_regret(results, samples, changes, horizon)

    ax1.plot(current_cum_regret["epsilon-Greedy"], label="stationary deterministic epsilon-Greedy", color="blue")
    ax1.legend(fontsize=15)
    ax1.grid()

    ax2 = fig.add_subplot(gs[0, 1])
    arm1_constant = [0] * horizon
    arm2_constant = [1] * horizon
    arm1_variable = [0] * (horizon // 2) + [1] * (horizon // 2)
    arm2_variable = [1] * (horizon // 2) + [0] * (horizon // 2)
    ax2.plot(arm1_constant, label="arm 1 stationary instance", color="blue")
    ax2.plot(arm2_constant, label="arm 2 stationary instance", color="blue", linestyle="--")
    ax2.set_xlabel("Rounds", fontsize=15)
    ax2.set_ylabel("Arm values", fontsize=15)
    ax2.set_title("Stationary Instance Details: Arms Values Over Time", fontsize=18)
    ax2.legend(fontsize=15)
    ax2.grid()

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(arm1_variable, label="arm 1 non-stationary instance", color="orange")
    ax3.plot(arm2_variable, label="arm 2 non-stationary instance", color="orange", linestyle="--")
    ax3.set_xlabel("Rounds", fontsize=15)
    ax3.set_ylabel("Arm values", fontsize=15)
    ax3.set_title("Non-Stationary Instance Details: Arms Values Over Time", fontsize=18)
    ax3.legend(fontsize=15)
    ax3.grid()

    plt.tight_layout()

    # Show plots
    plt.show()


def UCB_test():
    horizon = 10000
    ucb_radius_function = lambda t, c: np.sqrt((2 * np.log(horizon)) / (c))
    algo = UCB(2, ucb_radius_function)

    algorithms = {"UCBT": algo}
    changes = [(1, [0, 0]), ((horizon // 10) * 3 + 1, [0.2, 1])]

    results, samples = run_simulation(horizon, algorithms, changes, True)
    current_cum_regret = calculate_cumulant_regret(results, samples, changes, horizon)

    fig = plt.figure(figsize=(15, 7))

    # Define GridSpec
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[:, 0])
    ax1.plot(current_cum_regret["UCBT"], label="non-stationary UCBT", color="orange")
    ax1.set_xlabel("Rounds", fontsize=15)
    ax1.set_ylabel("Cumulative Regret", fontsize=15)
    ax1.set_title("Cumulative Regret of UCBT", fontsize=18)
    changes = [(1, [0.5, 0.55])]

    results, samples = run_simulation(horizon, algorithms, changes, True)
    current_cum_regret = calculate_cumulant_regret(results, samples, changes, horizon)

    ax1.plot(current_cum_regret["UCBT"], label="stationary UCBT", color="blue")
    ax1.legend(fontsize=15)
    ax1.grid()

    ax2 = fig.add_subplot(gs[0, 1])
    arm1_constant = [0.5] * horizon
    arm2_constant = [0.55] * horizon
    arm1_variable = [float(0)] * (horizon // 10 + 1) + [0.2] * (horizon - (horizon // 10 + 1))
    arm2_variable = [0] * (horizon // 10 + 1) + [1] * (horizon - (horizon // 10 + 1))
    ax2.plot(arm1_constant, label="arm 1 stationary instance", color="blue")
    ax2.set_ylim(0, 1)
    ax2.plot(arm2_constant, label="arm 2 stationary instance", color="blue", linestyle="--")
    ax2.set_xlabel("Rounds",  fontsize=15)
    ax2.set_ylabel("Arm values",  fontsize=15)
    ax2.set_title("Stationary Instance Details: Arms Values Over Time", fontsize=18)
    ax2.legend(fontsize=15)
    ax2.grid()

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(arm1_variable, label="arm 1 non-stationary instance", color="orange")
    ax3.plot(arm2_variable, label="arm 2 non-stationary instance", color="orange", linestyle="--")
    ax3.set_xlabel("Rounds",  fontsize=15)
    ax3.set_ylabel("Arm values",  fontsize=15)
    ax3.set_title("Non-Stationary Instance Details: Arms Values Over Time",  fontsize=18)
    ax3.legend(fontsize=15)
    ax3.grid()

    plt.tight_layout()

    # Show plots
    plt.show()


def swucb_test():
    N = 1
    horizon = 1000
    gamma = 10
    tau = np.floor(np.sqrt(horizon * np.log(horizon) / gamma)) * 2
    alpha = np.sqrt(2 * np.log(tau))
    c = np.floor(((tau ** 2) * (alpha ** 2) / 8) ** (1 / 3))
    delta = alpha / (c ** 0.5)
    sw_radius_function = lambda t, counter: np.sqrt((2 * np.log(min(t, tau))) / counter)
    sw_ucb = SWUCB(2, tau, sw_radius_function)
    algorithms = {"SWUCB": sw_ucb}
    changes = [(1, [0, 0])]
    # i = 0
    # while True:
    #     if i * tau + c + i + 1 > horizon:
    #         break
    #     changes.append((i * tau + c + i, [delta, 1]))
    #     if (i + 1) * tau + i + 1 > horizon:
    #         break
    #     changes.append(((i + 1) * tau + i + 1, [delta, 0]))
    #     i += 1
    # print(len(changes))
    for m in range(int(np.floor(gamma/2 - 1))+1):
        changes.append((m * tau + 2 * c + m + 1, [delta, 1]))
        changes.append(((m + 1) * tau + m, [delta, 0]))

    arm1_variable = []
    arm2_variable = []
    for i in range(len(changes)):
        cur_arm1 = changes[i][1][0]
        cur_arm2 = changes[i][1][1]
        if i < len(changes)-1:
            arm1_variable += [cur_arm1] * (int(changes[i+1][0] - changes[i][0]))
            arm2_variable += [cur_arm2] * (int(changes[i+1][0] - changes[i][0]))
        else:
            arm1_variable += [cur_arm1] * (int(horizon - changes[i][0] + 1))
            arm2_variable += [cur_arm2] * (int(horizon - changes[i][0] + 1))

    results, samples = run_simulation(horizon, algorithms, changes, True)
    current_cum_regret = calculate_cumulant_regret(results, samples, changes, horizon)
    fig = plt.figure(figsize=(15, 7))

    # Define GridSpec
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[:, 0])
    ax1.plot(current_cum_regret["SWUCB"], label="SW-UCB large tau instance", color="orange")
    ax1.set_xlabel("Rounds", fontsize=15)
    ax1.set_ylabel("Cumulative Regret", fontsize=15)
    ax1.set_title("Cumulative Regret of SW-UCB with Optimal Window Size", fontsize=18)

    delta_small_instance = np.sqrt(5 * np.log(tau) / (2*tau))
    changes = [(1, [0, delta_small_instance])]

    results, samples = run_simulation(horizon, algorithms, changes, True)
    current_cum_regret = calculate_cumulant_regret(results, samples, changes, horizon)

    ax1.plot(current_cum_regret["SWUCB"], label="SW-UCB small tau instance", color="blue")
    ax1.legend(fontsize=15)
    ax1.grid()

    ax2 = fig.add_subplot(gs[0, 1])
    arm1_constant = [0] * horizon
    arm2_constant = [delta_small_instance] * horizon

    ax2.set_ylim(0, 1)
    ax2.plot(arm1_constant, label="arm 1 stationary instance", color="blue")
    ax2.plot(arm2_constant, label="arm 2 stationary instance", color="blue", linestyle="--")
    ax2.set_xlabel("Rounds", fontsize=15)
    ax2.set_ylabel("Arm values", fontsize=15)
    ax2.set_title("Small tau Instance Details: Arms Values Over Time", fontsize=18)
    ax2.legend(fontsize=15)
    ax2.grid()

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(arm1_variable, label="arm 1 non-stationary instance", color="orange")
    ax3.plot(arm2_variable, label="arm 2 non-stationary instance", color="orange", linestyle="--")
    ax3.set_xlabel("Rounds", fontsize=15)
    ax3.set_ylabel("Arm values", fontsize=15)
    ax3.set_title("Large tau Instance Details: Arms Values Over Time", fontsize=18)
    ax3.legend(fontsize=15)
    ax3.grid()

    plt.tight_layout()

    # Show plots
    plt.show()


def DUCB_constant_instance():
    N = 1
    horizon = 10000
    gamma = np.linspace(start=0.95, stop=0.9999999, num=100)
    armpicks_ducb = []
    armpicks_swucb = []
    algo = DUCB(2, 2, 0.5)
    regrets_ducb = []
    regrets_swucb = []
    for g in tqdm(gamma):
        algo.discount_factor = g
        tau = 1+g+g**2
        sw_radius_function = lambda t, counter: np.sqrt((2 * np.log(min(t, tau))) / counter)
        sw_ucb = SWUCB(2, tau, sw_radius_function)
        # delta = np.sqrt(-np.log(1-g)*(1-g))
        delta = np.sqrt((5 * np.log(tau)) / (2 * tau))
        changes = [(1, [0, delta])]
        algorithms = {"DUCB": algo, "SWUCB": sw_ucb}
        results, samples = run_simulation(horizon, algorithms, changes, True)
        calc_arms = calculate_cumulant_armpicks(results, horizon, 2)
        armpicks_ducb.append(calc_arms["DUCB"][0][-1])
        armpicks_swucb.append(calc_arms["SWUCB"][0][-1])
        calc_regret = calculate_cumulant_regret(results, samples, changes, horizon)
        regrets_ducb.append(calc_regret["DUCB"][-1])
        regrets_swucb.append(calc_regret["SWUCB"][-1])
    plt.plot(gamma, armpicks_ducb, label="DUCB arm 0")
    plt.plot(gamma, armpicks_swucb, label="SWUCB arm 0")
    plt.axvline(x=1 - 0.25 * np.sqrt((1 / horizon)), color='red', linestyle='--', linewidth=0.8, label="Optimal Discount Factor")
    plt.xlabel("Discount Factor")
    plt.ylabel("Number of times suboptimal arm was picked")
    plt.title("DUCB on constant instance with different discount factors")
    plt.legend()
    plt.show()

    plt.plot(gamma, regrets_ducb, label="DUCB")
    plt.plot(gamma, regrets_swucb, label="SWUCB")
    plt.axvline(x=1 - 0.25 * np.sqrt((1 / horizon)), color='red', linestyle='--', linewidth=0.8, label="Optimal Discount Factor")
    plt.xlabel("Discount Factor")
    plt.ylabel("Cumulative Regret")
    plt.title("DUCB on constant instance with different discount factors")
    plt.legend()
    plt.show()


def DUCB_SWUCB_eqiv():
    N = 1
    horizon = 100000
    # gamma = 1 - 0.25 * np.sqrt((1 / horizon))
    gamma = 0.9
    ducb = DUCB(2, 2, gamma)
    win = (1-gamma**horizon)/(1-gamma)
    tau = 1/(1-gamma)
    sw_radius_function = lambda t, counter: np.sqrt((2 * np.log(min(t, tau))) / counter)
    delta = np.sqrt((5*np.log(win))/(2*win))
    sw_ucb = SWUCB(2, tau, sw_radius_function)
    algorithms = {"DUCB": ducb, "SWUCB": sw_ucb}
    changes = [(1, [0, delta])]
    cumulant_regret = {algo_name: np.zeros(horizon) for algo_name in algorithms.keys()}
    for i in range(N):
        results, samples = run_simulation(horizon, algorithms, changes, True)
        current_cum_regret = calculate_cumulant_regret(results, samples, changes, horizon)
        for algo_name in algorithms.keys():
            cumulant_regret[algo_name] += 1 / N * current_cum_regret[algo_name]

    print(cumulant_regret["DUCB"][-1], cumulant_regret["SWUCB"][-1])
    plt.plot(cumulant_regret["DUCB"], label="DUCB")
    plt.plot(cumulant_regret["SWUCB"], label="SWUCB")
    plt.legend()
    plt.show()



DUCB_SWUCB_eqiv()
