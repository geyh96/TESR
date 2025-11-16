# The science + ieee styles for IEEE papers: It is readable with black-white color mode.
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
list_NT = [300]
ideparture = 1
list_S = [2,3,4,5,6,7,8]


def result_with_input_info(The_DATA_MARK,the_folder):
    list_allfile = os.listdir(the_folder)
    list_file = []
    for ifile in list_allfile:
        # print(ifile)
        # print(fnmatch(ifile,The_DATA_MARK))
        if fnmatch(ifile,The_DATA_MARK):
            list_file.append(ifile)
    print(len(list_file))
    if len(list_file)!=100:
        print('some thing wrong in folder: {} with marker:{}'.format(the_folder,The_DATA_MARK))
        print('#######################################################################')
    the_res = []
    for idx_data in range(len(list_file)):
        with open(os.path.join(the_folder,  list_file[idx_data]), 'rb') as file:
            ff = pickle.load(file)
            the_res.append(ff)
    return  the_res, np.mean(the_res),np.std(the_res)


def get_data(the_folder,dicts,method):
    all_results = []
    for i in range(len(list_S)):
        # for j in range(len(list_Rotation)):
        # Power = list_Power[i]
        # Rotation = list_Rotation[j]
        ideparture = 1
        numS = list_S[i]
        # The_DATA_MARK = "*" + "_power_" + str(Power) + "_rotation_" + str(Rotation) + "_*"
        The_DATA_MARK = "*" + "_idep*_" + str(ideparture) + "_numS_" + str(numS) + "*"
        print(The_DATA_MARK)
        result,_,_ = result_with_input_info(The_DATA_MARK,the_folder)
        all_results.append(result)
    dicts[method] = all_results
    return dicts

colors= ['#FF0000','#66CDAA','#6495ED','#FFA07A','#BA55D3']
methods = ["TESR","DNN","DDR","TransIRM","SVM"]
folders = ["./Case_TESR/result/",
           "./Case_DNN/result/",
           "./Case_DDR/result/",
           "./Case_TransIRM/result/",
           "./Case_FT/result/"]
line_fmt = ["o-","v--","*-.","+:","s:"]
line_capsize = [10,5,5,5]
the_markers = ["o"]*5




dicts = {}



for i in range(5):
    dicts = get_data(folders[i],dicts,methods[i])




width = 0.3
fig = plt.figure(figsize=(24,12))
gs = gridspec.GridSpec(2, 2,height_ratios=[1,0.05])
axs = [plt.subplot(gs[i]) for i in range(2)]
positions = np.arange(1, len(list_S) + 1)
positions_nn = []

for i in range(1,len(list_S)+1):
    if i==1:
        positions_nn.append(i-width*2/3)
    else:
        positions_nn.append(i-width*2/3)

positions_dcor = []
for i in range(1,len(list_S)+1):
    if i==1:
        positions_dcor.append(i+width*2/3)
    else:
        positions_dcor.append(i+width*2/3)
positions_ft = []
for i in range(1,len(list_S)+1):
    if i==1:
        positions_ft.append(i)
    else:
        positions_ft.append(i)
bp_trans_dcor = axs[0].boxplot(dicts[methods[0]], positions=positions_dcor, widths=width*2/3, patch_artist=True, boxprops=dict(facecolor=colors[0]), medianprops=dict(color='#000000'), notch=True)

bp_dnn = axs[0].boxplot(dicts[methods[1]][0], positions=[positions[0]-width*4/3], widths=width*2/3, patch_artist=True, boxprops=dict(facecolor=colors[1]), medianprops=dict(color='#000000',linestyle='--'))

bp_dcor = axs[0].boxplot(dicts[methods[2]][0], positions=[positions[0]-2*width], widths=width*2/3, patch_artist=True, boxprops=dict(facecolor=colors[2]), medianprops=dict(color='#000000'))

bp_trans_dnn = axs[0].boxplot(dicts[methods[3]], positions=positions_nn, widths=width*2/3, patch_artist=True, boxprops=dict(facecolor=colors[3]), medianprops=dict(color='#000000',linestyle='--'), notch=True)

bp_dnn_ft = axs[0].boxplot(dicts[methods[4]], positions=positions_ft, widths=width*2/3, patch_artist=True, boxprops=dict(facecolor=colors[4]), medianprops=dict(color='#000000',linestyle=':'))
# bp_svm  = axs[0].boxplot(dicts[methods[4]][0], positions=[positions[0]-3*width], widths=width, patch_artist=True, boxprops=dict(facecolor='#BA55D3'), medianprops=dict(color='#000000'))
axs[0].set_title('($n_s, n_0, d, Type$) = (2000, 300, 60, I)', fontsize=24)
axs[0].set_xlabel('S', fontsize=24)
axs[0].set_ylabel('Accuracy',fontsize=24)
axs[0].tick_params(axis='both', which='major', labelsize=20)
axs[0].set_xticks(positions)
axs[0].set_xticklabels(list_S)
axs[0].grid(True,color = '#f0f0f0')  # 添加网格
axs[0].set_facecolor('white')  # 设置背景色
axs[0].set_ylim([0.4,0.85])




def get_data(the_folder,dicts,method):
    all_results = []
    for i in range(len(list_S)):
        # for j in range(len(list_Rotation)):
        # Power = list_Power[i]
        # Rotation = list_Rotation[j]
        ideparture =0
        numS = list_S[i]
        # The_DATA_MARK = "*" + "_power_" + str(Power) + "_rotation_" + str(Rotation) + "_*"
        The_DATA_MARK = "*" + "_ide*_" + str(ideparture) + "_numS_" + str(numS) + "*"

        print(The_DATA_MARK)
        result,_,_ = result_with_input_info(The_DATA_MARK,the_folder)

        all_results.append(result)
    dicts[method] = all_results
    return dicts

# list_Rotation = [0,15,30,45,60,75]
positions = np.arange(1, len(list_S) + 1)
dicts = {}

for i in range(5):
    dicts = get_data(folders[i],dicts,methods[i])
positions_nn = []
for i in range(1,len(list_S)+1):
    if i==1:
        positions_nn.append(i-width*2/3)
    else:
        positions_nn.append(i-2*width/3)
positions_dcor = []
for i in range(1,len(list_S)+1):
    if i==1:
        positions_dcor.append(i+width*2/3)
    else:
        positions_dcor.append(i+2*width/3)
positions_ft = []
for i in range(1,len(list_S)+1):
    if i==1:
        positions_ft.append(i)
    else:
        positions_ft.append(i)
# ['#FF0000','#FFA07A','#6495ED','#66CDAA','#BA55D3']
bp_trans_dcor = axs[1].boxplot(dicts[methods[0]], positions=positions_dcor, widths=width*2/3, patch_artist=True, boxprops=dict(facecolor=colors[0]), medianprops=dict(color='#000000'), notch=True)
bp_dnn = axs[1].boxplot(dicts[methods[1]][0], positions=[positions[0]-4*width/3], widths=width*2/3, patch_artist=True, boxprops=dict(facecolor=colors[1]), medianprops=dict(color='#000000',linestyle='--'))
bp_dcor = axs[1].boxplot(dicts[methods[2]][0], positions=[positions[0]-2*width], widths=width*2/3, patch_artist=True, boxprops=dict(facecolor=colors[2]), medianprops=dict(color='#000000'))
bp_trans_dnn = axs[1].boxplot(dicts[methods[3]], positions=positions_nn, widths=width*2/3, patch_artist=True, boxprops=dict(facecolor=colors[3]), medianprops=dict(color='#000000',linestyle='--'), notch=True)
bp_dnn_ft = axs[1].boxplot(dicts[methods[4]], positions=positions_ft, widths=width*2/3, patch_artist=True, boxprops=dict(facecolor=colors[4]), medianprops=dict(color='#000000'))
# bp_svm  = axs[1].boxplot(dicts[methods[4]][0], positions=[positions[0]-3*width], widths=width, patch_artist=True, boxprops=dict(facecolor='#BA55D3'), medianprops=dict(color='#000000'))
axs[1].set_title('($n_s, n_0, d, Type$) = (2000, 300, 60, II)', fontsize=24)
axs[1].set_xlabel('S', fontsize=24)
axs[1].set_ylabel('Accuracy',fontsize=24)
axs[1].tick_params(axis='both', which='major', labelsize=20)
axs[1].set_xticks(positions)
axs[1].set_xticklabels(list_S)
axs[1].grid(True,color = '#f0f0f0')  # 添加网格
axs[1].set_facecolor('white')  # 设置背景色
axs[1].set_ylim([0.4,0.85])


# plt.setp(axs, yticks=[0.6, 0.65, 0.7,0.75,0.8,0.85])
# plt.figlegend([bp_trans_dcor["boxes"][0],bp_trans_dnn["boxes"][0],bp_dcor["boxes"][0],bp_dnn["boxes"][0],bp_svm["boxes"][0]],
            #    ["TransRepre","TransDNN","DDR","DNN","SVM"], loc='lower center', bbox_to_anchor=(0.5, 0.05),ncol=5, prop={'size': 24})
plt.figlegend([bp_trans_dcor["boxes"][0],bp_dnn["boxes"][0],bp_dcor["boxes"][0],bp_trans_dnn["boxes"][0],bp_dnn_ft["boxes"][0]],
               ["TESR","DNN","DDR","TransIRM","FineTun"], loc='lower center', bbox_to_anchor=(0.5, 0.05),ncol=5, prop={'size': 24})
# plt.tight_layout()
plt.savefig('fig3.png',dpi=300,bbox_inches='tight')
plt.savefig('Simulation_Example3.pdf',bbox_inches='tight')

plt.show()

