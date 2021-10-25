import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_figure(x_data, y_data, name, xlabel, ylabel):
    fig, ax = plt.subplots()
    plt.setp(ax.spines.values(), linewidth=1.7)
    title = name
    ax.set_title(title, fontsize=17)
    ax.set_xlabel(xlabel, fontsize = 16)
    ax.set_ylabel(ylabel, fontsize = 16)

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    #ax.yaxis.set_tick_params(which='minor',direction="in")
    ax.tick_params(axis="x", direction="in", which='major', labelsize=13, width=2)
    ax.tick_params(axis="x", direction="in", which='minor', labelsize=13)

    ax.tick_params(axis="y", direction="in", which='major', labelsize=13, width=2)
    ax.tick_params(axis="y", direction="in", which='minor', labelsize=13)

    ax.plot(x_data, y_data)

    ax.set_xlim(xmin=0)
    ax.set_ylim(ymin=0)

    ax.grid()
    plt.legend(loc='upper right', ncol=1)
    plt.show()

class FRF:
    def __init__(self, filename):
        self.Freq = 0
        self.Amp  = 0
        self.real = 0
        self.imag = 0
        self.filename  = filename
        self.df = 0
        self.dis = 0
        self.max = 0
        self.index = 0

    def readExcel(self, axis):
        self.df = pd.read_excel(self.filename, header=None, names=None, index_col=None)
        if axis == "x":
            self.Freq = np.array(self.df.loc[0][3:])
            self.Amp = np.array(self.df.loc[4][3:])
            self.real = np.array(self.df.loc[2][3:])
            self.imag = np.array(self.df.loc[3][3:])
        else:
            self.Freq = np.array(self.df.loc[0][3:])
            self.Amp = np.array(self.df.loc[4][3:])
            self.real = np.array(self.df.loc[2][3:])
            self.imag = np.array(self.df.loc[3][3:])  

        W = self.Freq*2*np.pi
        self.dis = self.Amp/(W**2)
        stiff = 1/self.dis
        self.avg_Amp = moving_average(self.Amp, 10)
        self.avg_dis = moving_average(self.dis, 10)
        self.max = max(stiff)
        self.index = self.Freq[np.argmax(stiff)]
        real_dis = self.real/(W**2)
        imag_dis = self.imag/(W**2)
        # plot_figure(self.Freq, self.Amp, "FRF result", "freq", "amp")
            
def convert_to_stiffness(filename, axis):
    file_path = os.getcwd()
    file_path = file_path + "/test3"
    print("file path :", file_path)
    non_opt_max_list =[]
    opt_max_list =[]
    print("filename: ", filename)
    optimized_number  = 0
    for i in range(0, len(filename), 2):
        non_opt = FRF(file_path + filename[i])
        opt = FRF(file_path + filename[i+1])
        non_opt.readExcel(axis)
        opt.readExcel(axis)
        if opt.max >= non_opt.max:
            optimized_number+=1
        non_opt_max_list.append(non_opt.max)
        opt_max_list.append(opt.max)

        fig, ax = plt.subplots()
        plt.setp(ax.spines.values(), linewidth=1.7)
        title = "Impact hammer test Result of " + axis + str(1+round(i/2,1))
        ax.set_title(title, fontsize=17)
        ax.set_xlabel('Frequency [Hz]', fontsize = 16)
        ax.set_ylabel('Stiffness [$\mu$'+'m/N]', fontsize = 16)

        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')

        ax.scatter(non_opt.index, non_opt.max)
        ax.scatter(opt.index, opt.max)
        #ax.yaxis.set_tick_params(which='minor',direction="in")
        ax.tick_params(axis="x", direction="in", which='major', labelsize=13, width=2)
        ax.tick_params(axis="x", direction="in", which='minor', labelsize=13)

        ax.tick_params(axis="y", direction="in", which='major', labelsize=13, width=2)
        ax.tick_params(axis="y", direction="in", which='minor', labelsize=13)

        ax.plot(non_opt.Freq, 1/non_opt.dis, label='Non optimized')
        ax.plot(opt.Freq, 1/opt.dis, label='optimized')

        ax.set_xlim(xmin=0)
        # if axis =="y":
        #     ax.set_ylim(ymin=0, ymax=300000)
        # else:
        #     ax.set_ylim(ymin=0, ymax=6000000)

        ax.grid()
        plt.legend(loc='upper right', ncol=1)
        plt.show()
    print("optimzed along "+axis+"is :", optimized_number)
    
    fig, ax = plt.subplots()
    plt.setp(ax.spines.values(), linewidth=1.7)
    index = np.arange(5)
    width =0.27
    ax.set_title("Impact hammer test result" + axis, fontsize=17)
    ax.set_xlabel('Posture'  , fontsize = 16)
    ax.set_ylabel('Stiffness [$\mu$'+'m/N]', fontsize = 16)

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    #ax.yaxis.set_tick_params(which='minor',direction="in")
    ax.tick_params(axis="x", direction="in", which='major', labelsize=13, width=2)
    ax.tick_params(axis="x", direction="in", which='minor', labelsize=13)

    ax.tick_params(axis="y", direction="in", which='major', labelsize=13, width=2)
    ax.tick_params(axis="y", direction="in", which='minor', labelsize=13)
    rect1 = ax.bar(index, non_opt_max_list, width, color = 'r' )
    rect2 = ax.bar(index+width, opt_max_list, width, color = 'b' )
    ax.set_axisbelow(True)
    ax.grid()
    ax.legend( (rect1[0], rect2[0]), ('Non_optimized', 'optimized') )
    plt.show()
    return non_opt_max_list, opt_max_list

def compare_compliance(filename, axis):
    file_path = os.getcwd()
    file_path = file_path + "/test2"
    print("file path :", file_path)
    non_opt_max_list =[]
    opt_max_list =[]
    print("filename: ", filename)
    optimized_number  = 0
    for i in range(0, len(filename), 2):
        non_opt = FRF(file_path + filename[i])
        opt = FRF(file_path + filename[i+1])
        non_opt.readExcel(axis)
        opt.readExcel(axis)
        print("max: ", non_opt.max)
        if opt.max >= non_opt.max:
            print("optimized_number: ", optimized_number)
            optimized_number+=1
        non_opt_max_list.append(non_opt.max)
        opt_max_list.append(opt.max)

        fig, ax = plt.subplots()
        plt.setp(ax.spines.values(), linewidth=1.7)
        title = "Impact hammer test Result of " + axis + str(1+round(i/2,1))
        ax.set_title(title, fontsize=17)
        ax.set_xlabel('Frequency [Hz]', fontsize = 16)
        ax.set_ylabel('Compliance', fontsize = 16)

        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')

        #ax.yaxis.set_tick_params(which='minor',direction="in")
        ax.tick_params(axis="x", direction="in", which='major', labelsize=13, width=2)
        ax.tick_params(axis="x", direction="in", which='minor', labelsize=13)

        ax.tick_params(axis="y", direction="in", which='major', labelsize=13, width=2)
        ax.tick_params(axis="y", direction="in", which='minor', labelsize=13)
        # ax.set_xlim(xmin=0, xmax=200)
        # ax.set_ylim(ymin=0, ymax=0.0001)
        print(len(non_opt.avg_dis))
        print(len(non_opt.Freq)) 
        size_set = len(non_opt.avg_dis)
        ax.plot(non_opt.Freq[:size_set], non_opt.avg_Amp, label='Non optimized') #1024 - avg_size
        ax.plot(opt.Freq[:size_set], opt.avg_Amp, label='optimized')
        ax.grid()
        plt.legend(loc='upper right', ncol=1)
        plt.show()


if __name__ == "__main__":
    filename_y = []
    fileformat = ".xlsx"

    non_opt = "/npos3" + "_y" + fileformat
    opt = "/opos3"  + "_y" + fileformat
    filename_y.append(non_opt)
    filename_y.append(opt)

    non_opt = "/npos5" + "_y" + fileformat
    opt = "/opos5"  + "_y" + fileformat
    filename_y.append(non_opt)
    filename_y.append(opt)

    non_opt = "/npos8" + "_y" + fileformat
    opt = "/opos8"  + "_y" + fileformat
    filename_y.append(non_opt)
    filename_y.append(opt)

    non_opt = "/npos9" + "_y" + fileformat
    opt = "/opos9"  + "_y" + fileformat
    filename_y.append(non_opt)
    filename_y.append(opt)

    non_opt = "/npos10" + "_y" + fileformat
    opt = "/opos10"  + "_y" + fileformat
    filename_y.append(non_opt)
    filename_y.append(opt)

    ##non_y, opt_y = convert_to_stiffness(filename_y, "y")
    compare_compliance(filename_y, "y")
    filename_x = []

    non_opt = "/npos3" + "_x" + fileformat
    opt = "/opos3"  + "_x" + fileformat
    filename_x.append(non_opt)
    filename_x.append(opt)

    non_opt = "/npos5" + "_x" + fileformat
    opt = "/opos5"  + "_x" + fileformat
    filename_x.append(non_opt)
    filename_x.append(opt)

    non_opt = "/npos8" + "_x" + fileformat
    opt = "/opos8"  + "_x" + fileformat
    filename_x.append(non_opt)
    filename_x.append(opt)

    non_opt = "/npos9" + "_x" + fileformat
    opt = "/opos9"  + "_x" + fileformat
    filename_x.append(non_opt)
    filename_x.append(opt)

    non_opt = "/npos10" + "_x" + fileformat
    opt = "/opos10"  + "_x" + fileformat
    filename_x.append(non_opt)
    filename_x.append(opt)

    # non_x, opt_x = convert_to_stiffness(filename_x, "x")
    compare_compliance(filename_x, "x")
    # non_H, opt_H = [],[]
    # for i in range(len(non_y)):
    #     non_sum = non_x[i] + non_y[i]
    #     opt_sum = opt_x[i] + opt_y[i]
    #     fig, ax = plt.subplots()

    # plt.setp(ax.spines.values(), linewidth=1.7)
    # index = np.arange(5)
    # width =0.27

    # ax.set_title("Impact hammer test result(sum)" + axis, fontsize=17)
    # ax.set_xlabel('Posture'  , fontsize = 16)
    # ax.set_ylabel('Stiffness [$\mu$'+'m/N]', fontsize = 16)

    # ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    # ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # ax.xaxis.set_ticks_position('both')
    # ax.yaxis.set_ticks_position('both')
    # #ax.yaxis.set_tick_params(which='minor',direction="in")
    # ax.tick_params(axis="x", direction="in", which='major', labelsize=13, width=2)
    # ax.tick_params(axis="x", direction="in", which='minor', labelsize=13)

    # ax.tick_params(axis="y", direction="in", which='major', labelsize=13, width=2)
    # ax.tick_params(axis="y", direction="in", which='minor', labelsize=13)

    # rect1 = ax.bar(index, non_sum, width, color = 'r' )
    # rect2 = ax.bar(index+width, opt_sum, width, color = 'b' )
    
    # ax.set_axisbelow(True)
    # ax.grid()
    # ax.legend( (rect1[0], rect2[0]), ('Non_optimized', 'optimized') )
    # plt.show()
    
