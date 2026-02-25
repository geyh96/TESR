import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os
import pickle
from fnmatch import fnmatch

list_P = [60]
list_NS = [2000]
list_NT = [300]
dim_list = [8,16,32,64]
def result_with_input_info(The_DATA_MARK,the_folder):
    list_allfile = os.listdir(the_folder)
    list_file = []
    for ifile in list_allfile:
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
methods = ["TESR","DNN","DDR","TransIRM","FineTun"]
folders = [ "./Case_TESR/result/","./Case_DNN/result/",
           "./Case_DDR/result/","./Case_TransIRM/result/","./Case_FT/result/"]
line_fmt = ["o-","v--","*-.","+:","s:"]
dicts = {}
for i in range(5):
    dicts = get_data(folders[i],dicts,methods[i])

lines = []
labels = []
for i in range(5):
    print('method:',methods[i])
    print('means:',dicts[methods[i]]["mean"])
    print('stds:',dicts[methods[i]]["std"])
    


