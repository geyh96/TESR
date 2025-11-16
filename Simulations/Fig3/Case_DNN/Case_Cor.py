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
parser.add_argument('--P', type=int, default=30)
parser.add_argument('--numS', type=int, default=2)
parser.add_argument('--ideparture', type=int, default=0)
# parser.add_argument('--igroup', type=int, default=0)
line_args = parser.parse_args()
idx_data = line_args.iloop
NSource = line_args.NSource
NTarget = line_args.NTarget
P = line_args.P
numS = line_args.numS
ideparture = line_args.ideparture

m = numS
from itertools import product
args = class_args()
args.latent_dim = args.latent_dim*2

nsample = NTarget
NTest = args.NTest
The_val_ratio = 0.3
# NSource = args.NSource
# NTarget = args.NTarget
NSval= int(NSource * The_val_ratio)

NTval = int(NTarget * The_val_ratio)


ntest = NTest

print((NSource,m,NTarget))


args.Nnumber
print("Random Seed number")
print(args.seed + 1000*idx_data)
torch.manual_seed(args.seed + 1000*idx_data)
np.random.seed(args.seed +  1000*idx_data)


print("size of train, val, test: {:4d},{:4d}".format(ntest,NTval))
print("size of train, val, test: {:4d},{:4d}".format(ntest,NTval))


The_DATA_MARK = "id_" + str(idx_data) + "_idepar_" + str(ideparture) + "_numS_" + str(numS)


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
    DataS1_train = Get_data_0101(NSource, P, s=ss, is_power=ideparture, is_rotation=1-ideparture)
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
    DataS1_train = Get_data_0101(NSval, P, s=ss, is_power=ideparture, is_rotation=1-ideparture)
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
    DataT_train = Get_data_0101(NTarget,P,s=0, is_power=ideparture, is_rotation=1-ideparture)
    DataT_val = Get_data_0101(NTval,P,s=0, is_power=ideparture, is_rotation=1-ideparture)
    DataT_test = Get_data_0101(ntest,P,s=0, is_power=ideparture, is_rotation=1-ideparture)

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


nthres = 1

res_nthres = np.zeros(nthres)

loss_eval = np.ones(nthres) * 1e5
ithres = 0



for ithres in range(nthres):
    loss_best_ifold = 1e5


    net = Generator(xdim = P, ndim = args.latent_dim)
    optimizer = optim.RMSprop(net.parameters(),lr=args.lr_R, weight_decay=1e-4)

    the_dataset_train = my_regDataset(X=XT ,Rx=RxT,Y=YT,weight = np.ones_like(YT[:,0])+ 1e-6,setidx=setidxT)
    the_dataset_val = my_regDataset(X=XTval,Rx=RxTval,Y=YTval,weight = np.ones_like(YTval[:,0])+ 1e-6,setidx=setidxTval)
    Loader_train = DataLoader(the_dataset_train, batch_size=args.batch_size,shuffle=True)
    Loader_val = DataLoader(the_dataset_val, batch_size=len(the_dataset_val),shuffle=False)
    epoch = 1

    Loss = "BCE"
    for epoch in range(args.nEpochs_pred):
        net = net.train()
        if Loss == "MSE":
            MSEloss = nn.MSELoss(reduction='none')
        if Loss == "BCE":   
            MSEloss = nn.CrossEntropyLoss()

        for batch_idx, (X,Rx,Y,weight, setidx) in enumerate(Loader_train):

            X = torch.squeeze(X,dim=0).to(device)
            Y = torch.squeeze(Y,dim=0).to(device)
            Rx = torch.squeeze(Rx,dim=0).to(device)
            w, output = net(X) 
            YY = Y[:,0]
            loss = MSEloss(output,YY.long())  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        
        if Loss == "MSE":
            MSEloss = nn.MSELoss(reduction='none')
        if Loss == "BCE":
            MSEloss = nn.CrossEntropyLoss()
        net = net.eval()
        loss = 0
        with torch.no_grad():
            for X,Rx,Y, weight, setidx in Loader_val:
                X = torch.squeeze(X,dim=0)
                X = X.to(device)
                Y = torch.squeeze(Y,dim=0)
                Y = Y.to(device)
                w, output = net(X)
                YY = Y[:,0]
                lossii = MSEloss(output,YY.long())  
                loss = loss +  lossii
        loss = loss / len(Loader_val)
        loss_iepoch = loss
        if epoch % 10==0:
            print("Val"+'\nEpoch {}: predict_MSE: {:.4f}'.format(epoch, loss))
        if loss_iepoch < loss_best_ifold:
            print("Update Best Model In Epoch:{:4f} and val loss is :{:4f}".format(epoch,loss_iepoch))
            loss_best_ifold = loss_iepoch
            loss_eval[ithres] = loss_best_ifold
            torch.save(net.state_dict(), os.path.join(args.save,   The_DATA_MARK  + "Model_REweighted_iloop_" +  str(ithres) + '_Best' + '_net.pt'))

 
idx_best = np.argmin(loss_eval)
file_net_dict =  os.path.join(args.save,   The_DATA_MARK  + "Model_REweighted_iloop_" +  str(idx_best ) + '_Best' + '_net.pt')
net = Generator(xdim = P, ndim = args.latent_dim)
net.load_state_dict(torch.load(file_net_dict))




import pickle
x_test = torch.Tensor(XTtest)
with torch.no_grad():
    _,y_pred = net(x_test)


out = F.log_softmax(y_pred, dim=0)
y_indpred = torch.max(out ,1)[1]
y_indpred = torch.max(y_pred ,1)[1]



yy = y_indpred.int().numpy()
YTtest1 = YTtest[:,0].astype("int")

AAA = (yy==YTtest1).astype("int")
mse_pred = np.mean(AAA).copy()
print("The mse of prediction is {:4f}".format(mse_pred))
res_weight = mse_pred

import pickle
with open(os.path.join(args.save_pickle,  "Res_" + The_DATA_MARK  + '_Best' + '.pickle'), 'wb') as handle:
    pickle.dump(res_weight, handle, protocol=pickle.HIGHEST_PROTOCOL)


print("----------------------------------------------------------------------\n\n\n")

