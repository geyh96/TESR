

# The science + ieee styles for IEEE papers: It is readable with black-white color mode.
import matplotlib
# matplotlib.use("QtAgg")
matplotlib.use("agg")

import matplotlib.pyplot as plt
# import scienceplots
# plt.style.use(['science',"nature"])
import matplotlib.gridspec as gridspec
###First we read the result for all the methods.
import numpy as np
import pandas as pd
import os
import pickle
from fnmatch import fnmatch
def result_with_input_info(The_DATA_MARK,the_folder):
    list_allfile = os.listdir(the_folder)
    list_file = []
    for ifile in list_allfile:
        # print(ifile)
        # print(fnmatch(ifile,The_DATA_MARK))
        if fnmatch(ifile,The_DATA_MARK):
            list_file.append(ifile)

    print("len(list_file)")
    print(len(list_file))

    the_res = []
    for idx_data in range(len(list_file)):
        with open(os.path.join(the_folder,  list_file[idx_data]), 'rb') as file:
            ff = pickle.load(file)
            the_res.append(ff)

    return  the_res, np.mean(the_res),np.std(the_res)


def get_data(the_folder,dicts,method):
    all_means = []
    all_stds = []
    # for i in range(1):
    if 1:
        i = 0
        means = []
        stds = []
        for j in range(len(list_NS)):
            P = list_P[i]
            NS = list_NS[j]
            NT = list_NT[0] 
            The_DATA_MARK =  "*" +"_idx_Target_" + str(0) + "_NS_" + str(NS) + "_NT_" + str(NT) + "_P_" + str(P)+ "_*"
            print(The_DATA_MARK)
            _,mean,std = result_with_input_info(The_DATA_MARK,the_folder)
            means.append(mean)
            stds.append(std)
        all_means.append(means)
        all_stds.append(stds)
    dicts[method] = {"mean":all_means,"std":all_stds}
    return dicts



colors= ['#FF0000','#66CDAA','#6495ED','#FFA07A','#BA55D3']
methods = ["TESR","DNN","DDR","TransIRM","FineTun"]
folders = [ "./Case_TESR/result/","./Case_DNN/result/",
           "./Case_DDR/result/","./Case_TransIRM/result/","./Case_FT/result/"]
line_fmt = ["o-","v--","*-.","+:","s:"]
line_capsize = [10,5,5,5]

list_P = [60]


list_NS = [1000,2000,3000,4000,5000]
list_NS = [500,1000,1500,2000,2500]
list_NT = [300]
idxTarget = 0


dicts = {}
for i in range(5):
    dicts = get_data(folders[i],dicts,methods[i])
dicts

fig = plt.figure(figsize=(18,10))
gs = gridspec.GridSpec(2, 2,height_ratios=[1,0.1])
axs = [plt.subplot(gs[i]) for i in range(2)]
# fig, axs = plt.subplots(2, 3, figsize=(24,18))
lines = []
labels = []
for i in range(1):
    for j in range(5):
        P = list_P[i]
        NT = list_NT[0]
        means = dicts[methods[j]]["mean"][i]
        stds = dicts[methods[j]]["std"][i]
        line = axs[i].errorbar(list_NS, means, yerr=stds, fmt=line_fmt[j],color=colors[j],capsize=5,label=methods[j],linewidth=3)
        # line = axs[1+i].errorbar(list_NS, means, yerr=stds, fmt='o-',color=colors[j],capsize=5,label=methods[j],elinewidth=2)
        axs[i].set_title('$(Target, n_0,d) =$ (01,{},{})'.format(NT,P), fontsize=28)
        axs[i].tick_params(axis='both', which='major', labelsize=20)
        axs[i].set_xlabel(r'$n_s$', fontsize=24)
        axs[i].set_xticks(list_NS)
        axs[i].set_ylabel(r'Accuracy', fontsize=24)
        axs[i].grid(True,color = '#f0f0f0')  # 添加网格
        axs[i].set_facecolor('white')  # 设置背景色
        if i == 0:
            lines.append(line)
            labels.append(methods[j])        

def get_data1(the_folder,dicts,method):
    all_means = []
    all_stds = []
    for i in range(1):
        means = []
        stds = []
        for j in range(len(list_NS)):
            P = list_P[i]
            NS = list_NS[j]
            NT = list_NT[0] 
            The_DATA_MARK =  "*" +"_idx_Target_" + str(100) + "_NS_" + str(NS) + "_NT_" + str(NT) + "_P_" + str(P)+ "_*"
            _,mean,std = result_with_input_info(The_DATA_MARK,the_folder)
            means.append(mean)
            stds.append(std)
        all_means.append(means)
        all_stds.append(stds)
    dicts[method] = {"mean":all_means,"std":all_stds}
    return dicts

list_P = [60]

list_NS = [500,1000,1500,2000,2500]
list_NT = [300]
idxTarget = 100
dicts = {}
for i in range(5):
    dicts = get_data1(folders[i],dicts,methods[i])
# 调整子图布局

for i in range(1):
    for j in range(5):
        P = list_P[0]
        NT = list_NT[0]
        means = dicts[methods[j]]["mean"][i]
        stds = dicts[methods[j]]["std"][i]
        line = axs[1+i].errorbar(list_NS, means, yerr=stds, fmt=line_fmt[j],color=colors[j],capsize=5,label=methods[j],linewidth=3)
        # axs[1+i].set_title('$(N_S,P) =$({},{})'.format(NT,P), fontsize=28)
        axs[1+i].set_title('$(Target, n_0,d) =$ (02,{},{})'.format(NT,P), fontsize=28)
        axs[1+i].tick_params(axis='both', which='major', labelsize=20)
        axs[1+i].set_xlabel(r'$n_s$', fontsize=24)
        axs[1+i].set_xticks(list_NS)
        axs[1+i].set_ylabel(r'Accuracy', fontsize=24)
        axs[1+i].grid(True,color = '#f0f0f0')  # 添加网格
        axs[1+i].set_facecolor('white')  # 设置背景色
# plt.subplots_adjust(bottom=0.2)
plt.figlegend(lines, labels, loc='lower center',bbox_to_anchor=(0.5, 0.02),ncol=5, prop={'size': 32})
# plt.tight_layout(h_pad=2)
# plt.tight_layout()
# plt.show()
plt.savefig('Simulation_Example2.pdf',dpi=300)
