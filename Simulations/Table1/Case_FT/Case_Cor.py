###Regression2 Model(a)
import os
import sys
import numpy as np
import argparse


def mkdir(path):
    folder = os.path.exists(path)
    if not folder: #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path) #makedirs 创建文件时如果路径不存在会创建这个路径
        print("Done folder") 
    else:
        print("Folder Already")

# torch functions
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from torch.utils.data import DataLoader, Dataset

sys.path.append("..") 
from gen_data import *
from my_model import *
from my_energy import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--iloop', type=int, default=1)
parser.add_argument('--NSource', type=int, default=2000)
parser.add_argument('--NTarget', type=int, default=300)
parser.add_argument('--P', type=int, default=60)
parser.add_argument('--dim', type=int, default=8)
# parser.add_argument('--igroup', type=int, default=0)
line_args = parser.parse_args()
idx_data = line_args.iloop
NSource = line_args.NSource
NTarget = line_args.NTarget
P = line_args.P
dim = line_args.dim


from itertools import product
args = class_args()

#Only on target data
args.latent_dim = args.latent_dim*2


nsample = NTarget
NTest = args.NTest
The_val_ratio = 0.3
# NSource = args.NSource
# NTarget = args.NTarget
NSval= int(NSource * The_val_ratio)

NTval = int(NTarget * The_val_ratio)
m = args.m

ntest = NTest

print((NSource,m,NTarget))


args.Nnumber
print("Random Seed number")
print(args.seed + 1000*idx_data)
torch.manual_seed(args.seed + 1000*idx_data)
np.random.seed(args.seed +  1000*idx_data)


print("size of train, val, test: {:4d},{:4d}".format(ntest,NTval))
print("size of train, val, test: {:4d},{:4d}".format(ntest,NTval))


The_DATA_MARK = "idata_" + str(idx_data) + "_NS_" + str(NSource) + "_NT_" + str(NTarget) + "_P_" + str(P) + "_dim_" + str(dim)

mkdir("./result")
mkdir("./model")
# igroup = line_args.igroup
print("is the cuda avalable {:1d}".format(torch.cuda.is_available()))




print("Begin the args")
print( ',\n'.join(('{} = {}'.format(item, args.__dict__[item]) for item in args.__dict__)) )
print("End the args")



class my_regDataset(Dataset):
    def __init__(self, X, Rx, Y, weight, setidx):
        super().__init__()
        self.X = X
        self.Rx = Rx
        self.Y = Y
        self.weight = weight
        self.setidx = setidx
    
    def __getitem__(self, idx):
        X,Rx,Y,weight,setidx = self.X[idx], self.Rx[idx], self.Y[idx], self.weight[idx], self.setidx[idx]
        return X,Rx,Y,weight,setidx

    def __len__(self):
        return self.Y.shape[0]


def Totensor(x, device):
    return torch.Tensor(x).to(device)






########################################################################
list_XS = []
list_YS = []
list_setidxS = []
s=1
for s in range(m):
    ss = s + 1
    DataS1_train = Get_data_0101(NSource, P, s=ss)
    Xss = DataS1_train["X"]
    yss = DataS1_train["y"]
    Rss = DataS1_train["y"]
    setidxss = np.ones_like(yss)*ss
    list_XS.append(Xss)
    list_YS.append(yss)
    list_setidxS.append(setidxss)


XS = np.concatenate(list_XS,axis=0)
YS = np.concatenate(list_YS,axis=0)
setidxS = np.concatenate(list_setidxS,axis=0)


list_XSval = []
list_YSval = []
list_setidxSval = []
s=1
for s in range(0,m):
    ss = s + 1
    DataS1_train = Get_data_0101(NSval, P, s=ss)
    Xss = DataS1_train["X"]
    yss = DataS1_train["y"]
    Rss = DataS1_train["y"]
    setidxss = np.ones_like(yss)*ss
    list_XSval.append(Xss)
    list_YSval.append(yss)
    list_setidxSval.append(setidxss)



XSval = np.concatenate(list_XSval,axis=0)
YSval = np.concatenate(list_YSval,axis=0)
setidxSval = np.concatenate(list_setidxSval,axis=0)




##########################################################
if 1:
    s = 0
    DataT_train = Get_data_0101(NTarget,P,s=0)
    DataT_val = Get_data_0101(NTval,P,s=0)
    DataT_test = Get_data_0101(ntest,P,s=0)

    XT = DataT_train["X"]
    YT = DataT_train["y"]
    RxT = YT
    setidxT = np.ones_like(YT)*s

    XTval = DataT_val["X"]
    YTval = DataT_val["y"]
    RxTval = YTval
    setidxTval = np.ones_like(YTval)*s

    DataT_test.keys()
    XTtest = DataT_test["X"]
    YTtest = DataT_test["y"]
    Rxtest = YTtest
    setidxtest = np.ones_like(YTtest)*s

#############################################################################


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#########################################################################################################################################################################################################################################################################################################################################################################################################################################################################
print("begin the DNN modelling")







list_lambda_IRMloss = args.list_lambda_IRMloss_Source
nthres1 = len(list_lambda_IRMloss)
ithres = 0
loss_eval = np.ones(nthres1) * 1e5
for ithres in range(nthres1):
    loss_best_ifold = 1e5

    lambda_IRMloss = list_lambda_IRMloss[ithres]

    net = Generator(xdim = P, ndim = 2*dim,outdim = 1)
    optimizer = optim.RMSprop(net.parameters(),lr=args.lr_R, weight_decay=1e-6)
    net.to(device)

    the_dataset_train = my_regDataset(X=XS ,Rx=YS,Y=YS,weight = np.ones_like(YS[:,0])+ 1e-6,setidx=setidxS)
    the_dataset_val = my_regDataset(X=XSval,Rx=YSval,Y=YSval,weight = np.ones_like(YS[:,0])+ 1e-6,setidx=setidxSval)
    Loader_train = DataLoader(the_dataset_train, batch_size=args.batch_size,shuffle=True)
    Loader_val = DataLoader(the_dataset_val, batch_size=len(the_dataset_val),shuffle=False)
    epoch = 1
    for epoch in range(args.nEpochs):

        net = net.train()
        Loss = "MSE"
        if Loss == "MSE":
            MSEloss = nn.MSELoss()
        if Loss == "BCE":
            MSEloss = nn.BCEWithLogitsLoss()
        for batch_idx, (X,Rx,Y,weight, setidx) in enumerate(Loader_train):
            X = torch.squeeze(X,dim=0).to(device)
            Y = torch.squeeze(Y,dim=0).to(device)
            Rx = torch.squeeze(Rx,dim=0).to(device)
            w, output = net(X)
            loss = MSEloss(output,Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 1==0:
           
            #############################################################################################
            net = net.eval()
            if Loss == "MSE":
                MSEloss = nn.MSELoss()
            if Loss == "BCE":
                MSEloss = nn.BCEWithLogitsLoss()
            loss = 0
            with torch.no_grad():
                for batch_idx, (X,Rx,Y, weight, setidx) in enumerate(Loader_val):
                    X = torch.squeeze(X,dim=0)
                    X = X.to(device)
                    Y = torch.squeeze(Y,dim=0)
                    Y = Y.to(device)
                    weight = torch.unsqueeze(weight,dim=1).to(device)
                    setidx = torch.squeeze(setidx,dim=0).to(device)
                    w, output_im = net(X)
                    lossi = MSEloss(output_im,Y)  
                    loss = loss +  lossi
            loss_iepoch = loss / len(Loader_val)
            if epoch %10 == 0:
                print("Epoch {}: the Val loss {:4f}".format(epoch,loss_iepoch))
            #############################################################################################
            #############################################################################################
            if loss_iepoch < loss_best_ifold:
                print("Update Best Model In Epoch:{:4f} and val loss is :{:4f}".format(epoch,loss_iepoch))
                loss_best_ifold = loss_iepoch
                loss_eval[ithres] = loss_best_ifold
                torch.save(net.state_dict(), os.path.join(args.save, "dCOR_net_idata_" +  The_DATA_MARK  + "Mloop_" +  str(ithres) + '_net_Best' + '_net.pt'))
##########################prediction

##################################################################################################

idx_best = np.argmin(loss_eval)
print("best tuning:{:4f}".format(list_lambda_IRMloss[idx_best]))
file_net_dict =  os.path.join(args.save,  "dCOR_net_idata_" + The_DATA_MARK  + "Mloop_" +  str(idx_best) + '_net_Best' + '_net.pt')
net = Generator(xdim = P, ndim = 2*dim,outdim = 1)
net.load_state_dict(torch.load(file_net_dict))
net.to(device)

######################################################################################################################################

######################################################################################################################################

list_lambda_predloss = args.list_lambda_predloss
list_tuning = list(product(list_lambda_predloss))
nthres = len(list_tuning)

error_val_dCor2 = np.ones(len(list_tuning))*100

for ithres in range(nthres):
    print("ithres {:4f} in all nthres {:4f}".format(ithres,nthres))
    loss_best = 1e5
    lambda_pred = 1

    f1_pred_from_Rx = pred_from_Rx(2*dim,outdim=2)
    f1_pred_from_Rx.to(device)
    optimizerT = optim.RMSprop(
                [
                    {"params":f1_pred_from_Rx.parameters(),"lr":args.lr_R, "weight_decay":1e-6}
                    ]
                )
    lr_schedulerT = torch.optim.lr_scheduler.StepLR(optimizer=optimizerT,step_size=args.lr_step, gamma=args.decayRate)
    
    

    #########################################################################
    #########################################################################
    the_dataset_train = my_regDataset(X=XT,Rx=YT,Y=YT,weight=np.ones_like(YT[:,0])+ 1e-6,setidx=setidxT)
    the_dataset_val = my_regDataset(X=XTval,Rx=YTval,Y=YTval,weight=np.ones_like(YTval[:,0])+ 1e-6,setidx=setidxTval)
    Loader_train = DataLoader(the_dataset_train, batch_size=args.batch_size,shuffle=True)
    Loader_val = DataLoader(the_dataset_val, batch_size=len(the_dataset_val),shuffle=False)
    #########################################################################


    for epoch in range(args.nEpochs):
        f1_pred_from_Rx = f1_pred_from_Rx.train()
        net = net.eval()
        MSEloss = nn.CrossEntropyLoss()
        for batch_idx, (X,Rx,Y,weight, setidx) in enumerate(Loader_train):
            X = torch.squeeze(X,dim=0)
            Y = torch.squeeze(Y,dim=0)
            Rx = torch.squeeze(Rx,dim=0)
            setidx = torch.squeeze(setidx,dim=0)
            Xi, Yi, setidxi = X.to(device),Y.to(device),setidx.to(device)
            D = torch.randn(Yi.shape[0], 2*dim).to(device)
            w_prior, _ = net(Xi)
            # w, _ = net_Target(Xi)
            w_prior = w_prior.detach()
            Rx = w_prior
            Y_hat = f1_pred_from_Rx(Rx)
            YYi = Yi[:,0]
            loss = MSEloss(Y_hat,YYi.long())
            optimizerT.zero_grad()
            loss.backward()
            optimizerT.step()
        lr_schedulerT.step()
        if epoch % 1==0:
            net = net.eval()
            f1_pred_from_Rx = f1_pred_from_Rx.eval()
            loss = 0
            Loss = "BCE"
            if Loss == "MSE":
                MSEloss = nn.MSELoss()
            if Loss == "BCE":
                MSEloss = nn.CrossEntropyLoss()
            with torch.no_grad():
                for X,Rx,Y,weight, setidx in Loader_val:
                    X = torch.squeeze(X,dim=0)
                    X = X.to(device)
                    Y = torch.squeeze(Y,dim=0)
                    Y = Y.to(device)
                    Rx_est1,_ = net(X)
                    latent = Rx_est1
                    output = f1_pred_from_Rx(latent)
                    YY = Y[:,0]
                    lossi = MSEloss(output,YY.long()) 
                    loss = loss + lossi
            loss = loss / len(Loader_val)
            loss_iepoch = loss
            if epoch %10 ==0:
                print("Val"+'Epoch {}: predict_MSE: {:.4f}'.format(epoch, loss))
            ##################################################################
    
            if loss_iepoch < loss_best:
                error_val_dCor2[ithres] = loss_iepoch
                print("Update Best Model In Epoch:{:4f} with val loss PRED: {:4f}".format(epoch,loss_iepoch))
                loss_best = loss_iepoch
                torch.save(f1_pred_from_Rx.state_dict(), os.path.join(args.save, "dCORpred_idata_" + The_DATA_MARK  + "Mloop_"  + str(ithres) + '_net_Best'  + '_net.pt'))
                
######Testing in the ithres    
file_net_dict =  os.path.join(args.save,   "dCOR_net_idata_" + The_DATA_MARK  + "Mloop_" +  str(idx_best) + '_net_Best' + '_net.pt')
net = Generator(xdim = P, ndim = 2*dim,outdim = 1)
net.load_state_dict(torch.load(file_net_dict))

print("error_val_dCor2")
print(error_val_dCor2)
idx_best2 = np.argmin(error_val_dCor2)



file_net_dict = os.path.join(args.save, "dCORpred_idata_" + The_DATA_MARK  + "Mloop_" +str(idx_best2) + '_net_Best' + '_net.pt')
f1_pred_from_Rx = pred_from_Rx(2*dim, outdim = 2)
f1_pred_from_Rx.load_state_dict(torch.load(file_net_dict))


import pickle
x_test = torch.Tensor(XTtest)
with torch.no_grad():
    Rx_pred1,_ = net(x_test)
    Rx_pred = Rx_pred1
    y_pred = f1_pred_from_Rx(Rx_pred)

out = F.log_softmax(y_pred, dim=0)
y_indpred = torch.max(out ,1)[1]


yy = y_indpred.int().numpy()
YTtest1 = YTtest[:,0].astype("int")

AAA = (yy==YTtest1).astype("int")
mse_pred = np.mean(AAA).copy()
print("The mse of prediction is {:4f}".format(mse_pred))
res_weight = mse_pred

res = mse_pred

import pickle
with open(os.path.join(args.save_pickle,  "Result_" + The_DATA_MARK  + "_Mloop_" +  str(idx_data) + '_net_Best' + '_result.pickle'), 'wb') as handle:
    pickle.dump(
        res
        , handle, protocol=pickle.HIGHEST_PROTOCOL)
