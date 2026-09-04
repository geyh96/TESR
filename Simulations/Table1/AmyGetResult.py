# The science + ieee styles for IEEE papers: It is readable with black-white color mode.
import matplotlib
# matplotlib.use("QtAgg")
matplotlib.use("Agg")

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

list_P = [60]
list_NS = [2000]
# list_NT = [200, 400, 600, 800, 1000]
list_NT = [300]
dim_list = [8,16,32,64]
def result_with_input_info(The_DATA_MARK,the_folder):
    list_allfile = os.listdir(the_folder)
    list_file = []
    for ifile in list_allfile:
        # print(ifile)
        # print(fnmatch(ifile,The_DATA_MARK))
        if fnmatch(ifile,The_DATA_MARK):
            list_file.append(ifile)

    if len(list_file)!= 100:
        print("Warning: the number of files is not 100, but {}".format(len(list_file)))

    the_res = []
    for idx_data in range(len(list_file)):
        with open(os.path.join(the_folder,  list_file[idx_data]), 'rb') as file:
            ff = pickle.load(file)
            the_res.append(ff)

    return  the_res, np.mean(the_res),np.std(the_res)


def get_data(the_folder,dicts,method):
    all_means = []
    all_stds = []
    for dim in dim_list:
        The_DATA_MARK = "*" + "_NS_" + str(2000) + "_NT_" + str(300) + "_P_" + str(60) + "_dim_" + str(dim) + "_*"
        _,mean,std = result_with_input_info(The_DATA_MARK,the_folder)

        all_means.append(mean)
        all_stds.append(std)
    dicts[method] = {"mean":all_means,"std":all_stds}
    return dicts
colors= ['#FF0000','#66CDAA','#6495ED','#FFA07A','#BA55D3']
# methods = ["TransRep","TransDNN","DDR","DNN","SVM"]
methods = ["TESR","DNN","DDR","TransIRM","FineTun"]
folders = [ "./Case_TESR/result/","./Case_DNN/result/",
           "./Case_DDR/result/","./Case_TransIRM/result/","./Case_FT/result/"]

line_fmt = ["o-","v--","*-.","+:","s:"]


dicts = {}
for i in range(5):
    dicts = get_data(folders[i],dicts,methods[i])
# print(dicts['SVM'])
# fig, axs = plt.subplots(2, 3, figsize=(24,18))
lines = []
labels = []
for i in range(5):
    print('method:',methods[i])
    print('means:',dicts[methods[i]]["mean"])
    print('stds:',dicts[methods[i]]["std"])
# for i in range(3):
#     for j in range(5):
#         P = list_P[i]
#         NS = list_NS[0]
#         means = dicts[methods[j]]["mean"][i]
#         stds = dicts[methods[j]]["std"][i]
#         line = axs[i].errorbar(list_NT, means, yerr=stds, fmt=line_fmt[j],color=colors[j],capsize=5,label=methods[j],linewidth=3)
#         axs[i].set_title('$(n_s,d) =$ ({},{})'.format(NS,P), fontsize=28)
#         axs[i].tick_params(axis='both', which='major', labelsize=20)
#         axs[i].set_xlabel(r'$n_0$', fontsize=24)
#         axs[i].set_xticks(list_NT)
#         axs[i].set_ylabel(r'Accuracy', fontsize=24)
#         axs[i].grid(True,color = '#f0f0f0')  # 添加网格
#         axs[i].set_facecolor('white')  # 设置背景色
#         if i == 0:
#             lines.append(line)
#             labels.append(methods[j])        


