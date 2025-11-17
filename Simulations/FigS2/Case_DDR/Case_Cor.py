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
parser.add_argument('--iloop', type=int, default=7)
parser.add_argument('--NSource', type=int, default=1000)
parser.add_argument('--NTarget', type=int, default=200)
parser.add_argument('--P', type=int, default=20)
# parser.add_argument('--igroup', type=int, default=0)
line_args = parser.parse_args()
idx_data = line_args.iloop
NSource = line_args.NSource
NTarget = line_args.NTarget
P = line_args.P



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


The_DATA_MARK = "idata_" + str(idx_data) + "_NS_" + str(NSource) + "_NT_" + str(NTarget) + "_P_" + str(P)

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

if args.cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
#########################################################################################################################################################################################################################################################################################################################################################################################################################################################################
print("begin the DNN modelling")


list_lambda_Eloss = args.list_lambda_Eloss_Source
list_tuning = list_lambda_Eloss
nthres = len(list_tuning)

error_val_dCor = np.ones(nthres)
error_val_loss = np.ones(nthres)*100
ithres = 1

for ithres in range(nthres):


    print("ithres {:4f} in all nthres {:4f}".format(ithres,nthres))
    dCorloss_best = 0
    loss_best = 1e5
    the_tuning_params = list_tuning[ithres]
    lambda_Eloss = the_tuning_params
    args.lambda_Eloss = lambda_Eloss
    print("args.lambda_Eloss")
    print(lambda_Eloss)

    net = Generator(xdim = P, ndim = args.latent_dim)
    optimizer = optim.RMSprop(net.parameters(),lr=args.lr_R, weight_decay=1e-4)

    DCloss = Loss_DC()
    Eloss = Loss_Energy()


    the_dataset_train = my_regDataset(X=XT,Rx=YT,Y=YT,weight=np.ones_like(YT[:,0]).copy() + 1e-6,setidx=setidxT)
    the_dataset_val = my_regDataset(X=XTval,Rx=YTval,Y=YTval,weight=np.ones_like(YTval[:,0]).copy() + 1e-6,setidx=setidxTval)
    Loader_train = DataLoader(the_dataset_train, batch_size=args.batch_size,shuffle=True)
    Loader_val = DataLoader(the_dataset_val, batch_size=len(the_dataset_val),shuffle=False)
    epoch = 1
    for epoch in range(args.nEpochs):

        # train_NewDC_Energy(args, epoch, net,  Loader_train, optimizer, zlr,lambda_Eloss, DCloss,Eloss,Loss="BCE", device= device)
        ###########################################################
        ###########################################################
        net = net.train()
        Loss = "BCE"
        for batch_idx, (X,Rx,Y,weight, setidx) in enumerate(Loader_train):
            X = torch.squeeze(X,dim=0)
            Y = torch.squeeze(Y,dim=0)
            Rx = torch.squeeze(Rx,dim=0)
            Xi, Yi = X.to(device),Y.to(device)

            D = torch.randn(Yi.shape[0], args.latent_dim).to(device)
            w, _ = net(Xi)
            E_loss = Eloss(w,D)
            if Loss == "MSE":
                d_loss = DCloss(w, Yi.to(device))
            if Loss == "BCE":
                Y2 = 1 - Yi
                YY = torch.cat([Yi,Y2],dim=1)
                YY = YY + 1e-12
                d_loss = DCloss(w, YY.to(device))
            G_loss = lambda_Eloss * E_loss - d_loss 
            optimizer.zero_grad()
            G_loss.backward()
            optimizer.step()
        ###########################################################

        if epoch % 1==0:
            ###########################################################
            net = net.eval()
            dCor_loss = 0
            fit_Rx_loss = 0
            with torch.no_grad():
                for X,Rx,Y,weight, setidx in Loader_val:
                    X = torch.squeeze(X,dim=0)
                    X = X.to(device)
                    Y = torch.squeeze(Y,dim=0)
                    Rx = torch.squeeze(Rx,dim=0)
                    Y = Y.to(device)
                    Rx = Rx.to(device)
                    latent, output = net(X)
                    if Loss == "MSE":
                        dCor_loss += DCloss(latent, Y)
                    if Loss == "BCE":
                        Y2 = 1 - Y
                        YY = torch.cat([Y,Y2],dim=1)
                        YY = YY + 1e-12
                        dCor_loss += DCloss(latent, YY.to(device))
                    fit_Rx_loss += DCloss(latent, Rx)
            dCor_loss /= len(Loader_val)
            fit_Rx_loss /= len(Loader_val)
            if epoch % 10==0:
                print("On Val Dataset "+'\nEpoch {}: norm_latent: {:.4f},Test set: dCor_loss: {:.4f},Rx_loss: {:.4f}'.format(epoch,torch.norm(latent,p=2) ,dCor_loss,fit_Rx_loss))
            ###########################################################
            ###########################################################
            dCorloss_iepoch = dCor_loss
            if dCorloss_iepoch > dCorloss_best:
                print("Update Best Model In Epoch:{:4f} with val DC: {:4f}".format(epoch,dCorloss_iepoch))
                dCorloss_best = dCorloss_iepoch
                error_val_dCor[ithres] = dCorloss_best
                torch.save(net.state_dict(), os.path.join(args.save, "dCOR_idata_" + The_DATA_MARK  + "Mloop_" + str(ithres) + '_net_Best' + '_net.pt'))
    #####Train the Repre part down
idx_best = np.argmax(error_val_dCor)
file_net_dict =  os.path.join(args.save, "dCOR_idata_" + The_DATA_MARK  + "Mloop_"   + str(idx_best) + '_net_Best' + '_net.pt')
net = Generator(xdim = P, ndim = args.latent_dim)
net.load_state_dict(torch.load(file_net_dict))


f1_pred_from_Rx = pred_from_Rx(args.latent_dim,outdim = 2)
opt_pred1 = optim.RMSprop(f1_pred_from_Rx.parameters(),lr=args.lr_pred, weight_decay=1e-4)
lr_scheduler_pred = torch.optim.lr_scheduler.StepLR(optimizer=opt_pred1,step_size=args.lr_step, gamma=args.decayRate)

######################################################################################################################################
the_dataset_train = my_regDataset(X=XT,Rx=RxT,Y=YT,weight=np.ones_like(YT[:,0]).copy() + 1e-6,setidx=setidxT)
the_dataset_val = my_regDataset(X=XTval,Rx=RxTval,Y=YTval,weight=np.ones_like(YTval[:,0]).copy() + 1e-6,setidx=setidxTval)
Loader_train = DataLoader(the_dataset_train, batch_size=args.batch_size,shuffle=True)
Loader_val = DataLoader(the_dataset_val, batch_size=len(the_dataset_val),shuffle=False)
######################################################################################################################################


loss_best = 1e5
Loss = "BCE"
net = net.eval()
for epoch_2 in range(args.nEpochs_pred):
    ##########################################
    f1_pred_from_Rx = f1_pred_from_Rx.train()
    if Loss == "MSE":
        MSEloss = nn.MSELoss(reduction='none')
    if Loss == "BCE":
        MSEloss = nn.CrossEntropyLoss()
    for batch_idx, (X,Rx,Y, weight, setidx) in enumerate(Loader_train):
        X = torch.squeeze(X,dim=0)
        Y = torch.squeeze(Y,dim=0)
        Rx = torch.squeeze(Rx,dim=0)
        Xi, Yi = X.to(device),Y.to(device)
        Rx_est,_ = net(Xi)
        Rx_est = Rx_est.detach_()
        Yhat = f1_pred_from_Rx(Rx_est)
        YY = Y[:,0]
        loss = MSEloss(Yhat,YY.long()) 
        opt_pred1.zero_grad()
        loss.backward()
        opt_pred1.step()
    #######################################
    ########################################################
    lr_scheduler_pred.step()
    if epoch_2 %1 ==0:
        #######################
        f1_pred_from_Rx = f1_pred_from_Rx.eval()
        loss = 0
        if Loss == "MSE":
            MSEloss = nn.MSELoss(reduction='none')
        if Loss == "BCE":
            MSEloss = nn.CrossEntropyLoss()
        with torch.no_grad():
            for X,Rx,Y,weight, setidx in Loader_val:
                X = torch.squeeze(X,dim=0)
                X = X.to(device)
                Y = torch.squeeze(Y,dim=0)
                Y = Y.to(device)
                latent,_ = net(X)
                YY = Y[:,0]
                output = f1_pred_from_Rx(latent)

                lossi = MSEloss(output,YY.long()) 
                loss = loss + lossi
        loss = loss / len(Loader_val)
        if epoch_2 %10==0:
            print("Val" +'Epoch {}: predict_MSE: {:.4f}'.format(epoch_2, loss))
        loss_iepoch = loss
        ####################################
        if loss_iepoch < loss_best:
            error_val_loss[ithres] = loss_iepoch
            print("Update Best Model In Epoch:{:4f} with val loss PRED: {:4f}".format(epoch_2,loss_iepoch))
            loss_best = loss_iepoch
            torch.save(f1_pred_from_Rx.state_dict(), os.path.join(args.save, "dCORpred_idata_" + The_DATA_MARK + "Mloop_"  + str(idx_best) + '_net.pt'))


######Testing in the ithres
file_net_dict =  os.path.join(args.save, "dCOR_idata_" + The_DATA_MARK + "Mloop_"  + str(idx_best) + '_net_Best' + '_net.pt')
net = Generator(xdim = P, ndim = args.latent_dim)
net.load_state_dict(torch.load(file_net_dict))


file_net_dict = os.path.join(args.save, "dCORpred_idata_" + The_DATA_MARK  + "Mloop_"  + str(idx_best) + '_net.pt')
f1_pred_from_Rx = pred_from_Rx(args.latent_dim)
f1_pred_from_Rx.load_state_dict(torch.load(file_net_dict))


import pickle
x_test = torch.Tensor(XTtest)
with torch.no_grad():
    Rx_pred,_ = net(x_test)
    y_pred = f1_pred_from_Rx(Rx_pred)

out = F.log_softmax(y_pred, dim=0)
y_indpred = torch.max(out ,1)[1]


yy = y_indpred.int().numpy()
YTtest1 = YTtest[:,0].astype("int")

AAA = (yy==YTtest1).astype("int")

mse_pred_weightedmodel = np.mean(AAA).copy()
print("The mse of prediction is {:4f}".format(mse_pred_weightedmodel))

res = mse_pred_weightedmodel

import pickle
with open(os.path.join(args.save_pickle,  "dCOR_idata_result_" + The_DATA_MARK  + '_net_Best' + '_result.pickle'), 'wb') as handle:
    pickle.dump(
        res
        , handle, protocol=pickle.HIGHEST_PROTOCOL)
