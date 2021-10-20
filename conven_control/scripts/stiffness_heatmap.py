import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def heatmap_read(filename, axisname):
    # load_wb = load_workbook("C:/Users/UNIST/Desktop/stiffness_estimation/test_z.xlsx", data_only=True)
    df = pd.read_excel(filename, header=None, names=None, index_col=None)
    num_test = df.shape[0]
    print(df)
    xtick = [0.65, 0.8, 0.95]
    ytick = [-0.35, -0.2, -0.05, 0.1, 0.25]
    stiffness = []
    overall_stiff = []
    opt_stiffness = []
    overall_optstiff = []
    changed_stiffness_str = []
    changed_stiffness = []
    overal_change_str =[]
    overal_change = []
    for j in range(1, len(ytick)*len(xtick)+1):
        stiffness.append(round(df[2][j],5))
        opt_stiffness.append(round(df[3][j],5))
        temp = str(round(100*(df[3][j] - df[2][j])/df[2][j],2))
        temp2 = round(100*(df[3][j] - df[2][j])/df[2][j],2)
        temp = temp+"%"
        changed_stiffness_str.append(temp)
        changed_stiffness.append(temp2)
        if j%len(ytick)==0:
            overall_stiff.append(stiffness)
            overall_optstiff.append(opt_stiffness)
            overal_change_str.append(changed_stiffness_str)
            overal_change.append(changed_stiffness)
            stiffness = []
            opt_stiffness = []
            changed_stiffness_str = []
            changed_stiffness = []

    fig, ax = plt.subplots()
    im = ax.imshow(overall_stiff, vmin=1, vmax=6.5, cmap = 'autumn_r')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbarlabel = "stiffness"
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize = 12)
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(ytick)))
    ax.set_yticks(np.arange(len(xtick)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(ytick)
    ax.set_yticklabels(xtick)
    ax.set_xlabel('y (m)', fontsize = 13)
    ax.set_ylabel('x (m)', fontsize = 13)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    for i in range(len(xtick)):
        for j in range(len(ytick)):
                text = ax.text(j, i, overall_stiff[i][j],
                            ha="center", va="center", color="black")
    ax.set_title("Stiffness of Non-optimized posture(" + axisname+ " axis)", fontsize = 16)
    fig.tight_layout()
    plt.show()


    fig, ax = plt.subplots()
    im = ax.imshow(overall_optstiff, vmin=1, vmax=6.5,   cmap = 'autumn_r')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbarlabel = "stiffness"
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize = 12)
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(ytick)))
    ax.set_yticks(np.arange(len(xtick)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(ytick)
    ax.set_yticklabels(xtick)
    ax.set_xlabel('y (m)', fontsize = 13)
    ax.set_ylabel('x (m)', fontsize = 13)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    for i in range(len(xtick)):
        for j in range(len(ytick)):
                text = ax.text(j, i, overall_optstiff[i][j],
                            ha="center", va="center", color="black")
    ax.set_title("Stiffness of optimized posture(" + axisname+ " axis)", fontsize = 16)
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    im = ax.imshow(overal_change, vmin=-100, vmax=500,   cmap = 'autumn_r')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbarlabel = "ratio of changed stiffness"
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize = 12)
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(ytick)))
    ax.set_yticks(np.arange(len(xtick)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(ytick)
    ax.set_yticklabels(xtick)
    ax.set_xlabel('y (m)', fontsize = 13)
    ax.set_ylabel('x (m)', fontsize = 13)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    for i in range(len(xtick)):
        for j in range(len(ytick)):
            if overal_change[i][j] < 0:
                text = ax.text(j, i, overal_change_str[i][j],
                        ha="center", va="center", color="Red", fontweight="bold")
            else:
                text = ax.text(j, i, overal_change_str[i][j],
                            ha="center", va="center", color="black")
    ax.set_title("Ratio of changed stiffness(" + axisname+ " axis)", fontsize = 16)
    fig.tight_layout()
    plt.show()
if __name__ == "__main__":
    filename = ["../data/x_stiffness_compare.xlsx", "../data/y_stiffness_compare.xlsx", "../data/z_stiffness_compare.xlsx"]
    axisname = ["x", "y", "z"]
    for i in range(3):
        heatmap_read(filename[i], axisname[i])
