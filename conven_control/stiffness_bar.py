import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

def stiffness_figure(opt_file, nonopt_file):
    # load_wb = load_workbook("C:/Users/UNIST/Desktop/stiffness_estimation/test_z.xlsx", data_only=True)
    opt_df = pd.read_excel(opt_file, header=None, names=None, index_col=None)
    nonopt_df = pd.read_excel(nonopt_file, header=None, names=None, index_col=None)
    num_posture = len(opt_df[0])-1
    print("number of tested postuer: ", num_posture)
    print("stiffness compare figure drawing...")
    print(opt_df[9][0])
    opt_x_stiffness = opt_df[9][1:]
    print(opt_df[10][0])
    opt_y_stiffness = opt_df[10][1:]
    print(opt_df[11][0])
    opt_z_stiffness = opt_df[11][1:]
    print(nonopt_df[9][0])
    nonopt_x_stiffness = nonopt_df[9][1:]
    print(nonopt_df[10][0])
    nonopt_y_stiffness = nonopt_df[10][1:]
    print(nonopt_df[11][0])
    nonopt_z_stiffness = nonopt_df[11][1:]
    data = [nonopt_x_stiffness, opt_x_stiffness, nonopt_y_stiffness, opt_y_stiffness, nonopt_z_stiffness, opt_z_stiffness]
    not_opt_posture_number_x =0
    not_opt_posture_number_y =0
    not_opt_posture_number_z =0
    deviation_num = 0
    print("here", nonopt_x_stiffness[1])
    for i in range(1, num_posture+1):
        deviation = opt_df[8][i]-nonopt_df[8][i]
        if deviation > 0:
            deviation_num+=1
        diff_x = opt_x_stiffness[i]-nonopt_x_stiffness[i]
        if diff_x<0:
            not_opt_posture_number_x+=1

        diff_y = opt_y_stiffness[i]-nonopt_y_stiffness[i]
        if diff_y<0:
            not_opt_posture_number_y+=1

        diff_z = opt_z_stiffness[i]-nonopt_z_stiffness[i]
        if diff_z<0:
            not_opt_posture_number_z+=1
    print(deviation_num, not_opt_posture_number_x, not_opt_posture_number_y, not_opt_posture_number_z, "out of", num_posture)

    for i in range(0,6,2):
        fig, ax = plt.subplots()
        plt.setp(ax.spines.values(), linewidth=1.7)
        index = np.arange(num_posture)
        width =0.27

        ax.set_title("Non optimized vs optimized stiffness", fontsize=17)
        ax.set_xlabel('Posture'  , fontsize = 16)
        ax.set_ylabel('Stiffness', fontsize = 16)

        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        #ax.yaxis.set_tick_params(which='minor',direction="in")
        ax.tick_params(axis="x", direction="in", which='major', labelsize=13, width=2)
        ax.tick_params(axis="x", direction="in", which='minor', labelsize=13)

        ax.tick_params(axis="y", direction="in", which='major', labelsize=13, width=2)
        ax.tick_params(axis="y", direction="in", which='minor', labelsize=13)
        print(i, i+1)
        rect1 = ax.bar(index, data[i], width, color = 'r' )
        rect2 = ax.bar(index+width, data[i+1], width, color = 'b' )

        ax.legend( (rect1[0], rect2[0]), ('Non_optimized', 'optimized') )
        plt.show()



if __name__ == "__main__":
    filename = ["optimized_result.xlsx", "new_non_optimized_result.xlsx"]
    stiffness_figure(filename[0], filename[1])