import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

# from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


def stiffness_figure(opt_file, nonopt_file):
    opt_file = "../data/" + opt_file
    nonopt_file = "../data/" + nonopt_file

    opt_df = pd.read_excel(opt_file, header=None, names=None, index_col=None)
    nonopt_df = pd.read_excel(nonopt_file, header=None, names=None, index_col=None)
    num_posture = len(opt_df[0]) - 1
    print("number of tested postuer: ", num_posture)
    print("stiffness compare figure drawing...")

    deviation_num = 0
    deviation_data = [nonopt_df[10][1:], opt_df[10][1:]]

    for i in range(1, num_posture + 1):
        deviation = opt_df[10][i] - nonopt_df[10][i]
        if deviation < 0:
            deviation_num += 1

    print(deviation_num, "out of", num_posture, "is optimized")

    fig, ax = plt.subplots()
    plt.setp(ax.spines.values(), linewidth=1.7)
    index = np.arange(num_posture)
    width = 0.27

    ax.set_title("Cartesian Deviation by External Force", fontsize=17)
    ax.set_xlabel("Posture", fontsize=16)
    ax.set_ylabel("Deviation", fontsize=16)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    # ax.yaxis.set_tick_params(which='minor',direction="in")
    ax.tick_params(axis="x", direction="in", which="major", labelsize=13, width=2)
    ax.tick_params(axis="x", direction="in", which="minor", labelsize=13)

    ax.tick_params(axis="y", direction="in", which="major", labelsize=13, width=2)
    ax.tick_params(axis="y", direction="in", which="minor", labelsize=13)
    rect1 = ax.bar(index + 1, deviation_data[0], width, color="r")
    rect2 = ax.bar(index + 1 + width, deviation_data[1], width, color="b")

    ax.set_axisbelow(True)
    ax.grid(color="#A2A6AB", axis="y")

    ax.legend((rect1[0], rect2[0]), ("Non_optimized", "optimized"))
    plt.show()


if __name__ == "__main__":
    filename = ["opt_curve2_v1.xlsx", "ros_curve2_v1.xlsx"]
    stiffness_figure(filename[0], filename[1])
