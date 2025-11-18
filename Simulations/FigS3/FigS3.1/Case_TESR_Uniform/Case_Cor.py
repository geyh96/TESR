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


####################################################################################################################################################################################################################################################################################################################################################################################################################################################


print("begin the DNN modelling")
print( ',\n'.join(('{} = {}'.format(item, args.__dict__[item]) for item in args.__dict__)) )
list_lambda_Iloss = args.list_lambda_Invarloss_Source 
list_lambda_Eloss = args.list_lambda_Eloss_Source
list_tuning = list(product(list_lambda_Iloss,list_lambda_Eloss))
nthres1 = len(list_tuning)


error_val_dCor = np.zeros(nthres1)


ithres = 0


for ithres in range(nthres1):
    print("ithres:{:4f}".format(ithres))
    dCorloss_best = 1e5
    loss_best = 1e5

    lambda_Iloss = 1
    lambda_Eloss = 1



    net = Generator(xdim = P, ndim = args.latent_dim, outdim = 2)
    optimizer = optim.RMSprop(net.parameters(),lr=args.lr_R, weight_decay=1e-4)
    
    DCloss = Loss_DC()
    Eloss = Loss_Energy()

    #########################################################################
    #########################################################################
    the_dataset_train = my_regDataset(X=XS,Rx=YS,Y=YS,weight=np.ones_like(YS[:,0])+ 1e-6,setidx=setidxS)
    the_dataset_val = my_regDataset(X=XSval,Rx=YSval,Y=YSval,weight=np.ones_like(YSval[:,0])+ 1e-6,setidx=setidxSval)
    Loader_train = DataLoader(the_dataset_train, batch_size=args.batch_size,shuffle=True)
    Loader_val = DataLoader(the_dataset_val, batch_size=len(the_dataset_val),shuffle=False)

    epoch = 1
    for epoch in range(args.nEpochs):
        Loss = "MSE"
        net = net.train()
        for batch_idx, (X,Rx,Y,weight, setidx) in enumerate(Loader_train):
            X = torch.squeeze(X,dim=0)
            Y = torch.squeeze(Y,dim=0)
            Rx = torch.squeeze(Rx,dim=0)
            setidx = torch.squeeze(setidx,dim=0)
            Xi, Yi, setidxi = X.to(device),Y.to(device),setidx.to(device)
            D = torch.rand(Yi.shape[0], args.latent_dim).to(device)
            w, _ = net(Xi)

            E_loss = Eloss(w,D)
            list_cclass = []
            for im in range(m):
                iim = im + 1
                list_cclass.append((setidx==iim).float() +1e-6)
    

            cclass = torch.cat(list_cclass,dim=1)
            inva_loss = DCloss(w, cclass.to(device))
            ##################class neutral
            d_loss = 0
            if Loss == "MSE":
                for im in range(m):
                    iim = im + 1
                    ind_iim = torch.where(setidx==iim)[0]
                    d_loss_iim = DCloss(w[ind_iim,:], Yi[ind_iim,:].to(device))
                    d_loss = d_loss + d_loss_iim
            G_loss = lambda_Eloss * E_loss - d_loss + lambda_Iloss * inva_loss
            optimizer.zero_grad()
            G_loss.backward()
            optimizer.step()


        if epoch % 1==0:
            ############################################################
            net = net.eval()
            dCor_loss = 0
            fit_Rx_loss = 0
            with torch.no_grad():
                for X,Rx,Y,weight, setidx in Loader_val:
                    X = torch.squeeze(X,dim=0)
                    Y = torch.squeeze(Y,dim=0)
                    Rx = torch.squeeze(Rx,dim=0)
                    setidx = torch.squeeze(setidx,dim=0)
                    Xi, Yi, setidxi = X.to(device),Y.to(device),setidx.to(device)
                    D = torch.rand(Yi.shape[0], args.latent_dim).to(device)
                    w, _ = net(Xi)

                    E_loss = Eloss(w,D)
                    list_cclass = []
                    for im in range(m):
                        iim = im + 1
                        list_cclass.append((setidx==iim).float() +1e-6)
            

                    cclass = torch.cat(list_cclass,dim=1)
                    inva_loss = DCloss(w, cclass.to(device))
                    ##################class neutral
                    d_loss = 0
                    if Loss == "MSE":
                        for im in range(m):
                            iim = im + 1
                            ind_iim = torch.where(setidx==iim)[0]
                            d_loss_iim = DCloss(w[ind_iim,:], Yi[ind_iim,:].to(device))
                            d_loss = d_loss + d_loss_iim
                    G_loss = lambda_Eloss * E_loss - d_loss + lambda_Iloss * inva_loss
                    dCor_loss += G_loss.item()
            dCor_loss /= len(Loader_val)
            if epoch % 10==0:
                print('\nEpoch {}: Test set: dCor_loss: {:.4f}'.format(epoch, dCor_loss))
            dCorloss_iepoch = dCor_loss
            ############################################################
            ############################################################
            ############################################################
            if dCorloss_iepoch < dCorloss_best:
                print("Update Best Model In Epoch:{:4f} with val DC: {:4f}".format(epoch,dCorloss_iepoch))
                dCorloss_best = dCorloss_iepoch
                error_val_dCor[ithres] = dCorloss_best
                torch.save(net.state_dict(), os.path.join(args.save, "dCOR_idata_" + The_DATA_MARK  + "Mloop_" +str(ithres) + '_net_Best' + '_net.pt'))
    print("ithres:{:4f} end".format(ithres))



##################################################################################################
print("error_val_dCor")
print(error_val_dCor)


idx_best = np.argmax(error_val_dCor)

file_net_dict =  os.path.join(args.save, "dCOR_idata_" + The_DATA_MARK  + "Mloop_"  + str(idx_best) + '_net_Best' + '_net.pt')
net = Generator(xdim = P, ndim = args.latent_dim, outdim = 2)
net.load_state_dict(torch.load(file_net_dict))

######################################################################################################################################
######################################################################################################################################


ithres = 0
Loss = "BCE"
list_lambda_predloss = args.list_lambda_predloss
list_lambda_priorloss = args.list_lambda_priorloss
list_lambda_Eloss = args.list_lambda_Eloss
list_tuning = list(product(list_lambda_predloss,list_lambda_priorloss,list_lambda_Eloss))
nthres2 = len(list_tuning)
# nthres2=5
error_val_dCor2 = np.zeros(nthres2)
for ithres in range(nthres2):
    print("ithres {:4f} in all nthres {:4f}".format(ithres,len(list_tuning)))
    dCorloss_best = 1e5


    the_tuning_params = list_tuning[ithres]
    lambda_pred = 1
    lambda_prior = 1
    lambda_Eloss = 1
    print("list_lambda_predloss")
    print([lambda_pred,lambda_prior,lambda_Eloss])

    net_Target = Generator(xdim = P, ndim = args.latent_dim,outdim = 2)
    file_net_dict =  os.path.join(args.save, "dCOR_idata_" + The_DATA_MARK  + "Mloop_"  + str(idx_best) + '_net_Best' + '_net.pt')
    net_Target.load_state_dict(torch.load(file_net_dict))

    optimizerT = optim.RMSprop(
                [
                    {"params":net_Target.parameters(),"lr":args.lr_R, "weight_decay":1e-4},
                    ]
                )

    DCloss = Loss_DC()
    Eloss = Loss_Energy()
    
    #########################################################################
    #########################################################################
    the_dataset_train = my_regDataset(X=XT,Rx=YT,Y=YT,weight=np.ones_like(YT[:,0])+ 1e-6,setidx=setidxT)
    the_dataset_val = my_regDataset(X=XTval,Rx=YTval,Y=YTval,weight=np.ones_like(YTval[:,0])+ 1e-6,setidx=setidxTval)
    Loader_train = DataLoader(the_dataset_train, batch_size=args.batch_size,shuffle=True)
    Loader_val = DataLoader(the_dataset_val, batch_size=len(the_dataset_val),shuffle=False)

    #########################################################################
    #########################################################################
    epoch = 1
    loss_best = 1e5

    for epoch in range(args.nEpochs):
        net_Target = net_Target.train()
        net = net.eval()
        for batch_idx, (X,Rx,Y,weight, setidx) in enumerate(Loader_train):
            X = torch.squeeze(X,dim=0)
            Y = torch.squeeze(Y,dim=0)
            Rx = torch.squeeze(Rx,dim=0)
            Xi, Yi, setidxi = X.to(device),Y.to(device),setidx.to(device)
            D = torch.rand(Yi.shape[0], args.latent_dim).to(device)
            w_prior, _ = net(Xi)
            w, _ = net_Target(Xi)
            w_prior = w_prior.detach()
            Rx = torch.cat((w_prior,w),dim=1)
            D_loss = DCloss(Yi, Rx)
            E_loss = Eloss(w,D)
            d_loss_prior = DCloss(w, w_prior)
            G_loss =  -1*D_loss  +  lambda_prior* d_loss_prior  + lambda_Eloss*E_loss
            optimizerT.zero_grad()
            G_loss.backward()
            optimizerT.step()
        if epoch % 1==0:
            net = net.eval()
            net_Target = net_Target.eval()
            loss = 0
            with torch.no_grad():
                for X,Rx,Y,weight, setidx in Loader_val:
                    X = torch.squeeze(X,dim=0)
                    Y = torch.squeeze(Y,dim=0)
                    Rx = torch.squeeze(Rx,dim=0)
                    Xi, Yi, setidxi = X.to(device),Y.to(device),setidx.to(device)
                    D = torch.rand(Yi.shape[0], args.latent_dim).to(device)
                    w_prior, _ = net(Xi)
                    w, _ = net_Target(Xi)
                    w_prior = w_prior.detach()
                    Rx = torch.cat((w_prior,w),dim=1)
                    D_loss = DCloss(Yi, Rx)
                    E_loss = Eloss(w,D)
                    d_loss_prior = DCloss(w, w_prior)
                    G_loss =  -1*D_loss  +  lambda_prior* d_loss_prior  + lambda_Eloss*E_loss
                    loss += G_loss.item()
            loss = loss / len(Loader_val)
            loss_iepoch = loss
            if epoch %10==0:
                print("Val" +'Epoch {}: predict_MSE: {:.4f}'.format(epoch, loss))
            ###########################################

            if loss_iepoch < loss_best:
                error_val_dCor2[ithres] = loss_iepoch
                print("Update Best Model In Epoch:{:4f} with val pdCor: {:4f}".format(epoch,loss_iepoch))
                loss_best = loss_iepoch
                torch.save(net_Target.state_dict(), os.path.join(args.save, "dCOR_net2_idata_" + The_DATA_MARK + "Mloop_" +str(ithres) + '_net_Best' + '_net.pt'))




######Testing in the ithres    
file_net_dict =  os.path.join(args.save, "dCOR_idata_" + The_DATA_MARK  + "Mloop_"  + str(idx_best) + '_net_Best' + '_net.pt')
net = Generator(xdim = P, ndim = args.latent_dim,outdim =  2)
net.load_state_dict(torch.load(file_net_dict))



print("error_val_dCor2")
print(error_val_dCor2)
idx_best2 = np.argmax(error_val_dCor2)
print("list_tuning[idx_best2]")
print(list_tuning[idx_best2])
file_net_dict =  os.path.join(args.save, "dCOR_net2_idata_" + The_DATA_MARK  + "Mloop_"  + str(idx_best2) + '_net_Best' + '_net.pt')
net2 = Generator(xdim = P, ndim = args.latent_dim,outdim =  2)
net2.load_state_dict(torch.load(file_net_dict))





#########################################Training a Learner to evaluate the performance of the methods
net = net.eval()
net2 = net2.eval()
f1_pred_from_Rx = pred_from_Rx(args.latent_dim*2,outdim=2)

optimizer_f1 = optim.RMSprop(
            [
                {"params":f1_pred_from_Rx.parameters(),"lr":args.lr_R, "weight_decay":1e-4}
                ]
            )

the_dataset_train = my_regDataset(X=XT,Rx=RxT,Y=YT,weight=np.ones_like(YT[:,0])+ 1e-6,setidx=setidxT)
the_dataset_val = my_regDataset(X=XTval,Rx=RxTval,Y=YTval,weight=np.ones_like(YTval[:,0])+ 1e-6,setidx=setidxTval)
Loader_train = DataLoader(the_dataset_train, batch_size=args.batch_size,shuffle=True)
Loader_val = DataLoader(the_dataset_val, batch_size=len(the_dataset_val),shuffle=False)

loss_best = 1e5

MSEloss = nn.CrossEntropyLoss()
for epoch in range(args.nEpochs_pred):
    f1_pred_from_Rx = f1_pred_from_Rx.train()
    for batch_idx, (X,Rx,Y,weight, setidx) in enumerate(Loader_train):
        X = torch.squeeze(X,dim=0)
        Y = torch.squeeze(Y,dim=0)
        Rx = torch.squeeze(Rx,dim=0)
        Xi, Yi, setidxi = X.to(device),Y.to(device),setidx.to(device)
        w1, _ = net(Xi)
        w2, _ = net2(Xi)
        w1 = w1.detach()
        w2 = w2.detach()
        Rx = torch.cat((w1,w2),dim=1)
        Yhat = f1_pred_from_Rx(Rx)
        G_loss = MSEloss(Yhat,Yi[:,0].long() )
        optimizer_f1.zero_grad()
        G_loss.backward()
        optimizer_f1.step()
    if epoch % 1==0:
        net = net.eval()
        net_Target = net_Target.eval()
        f1_pred_from_Rx = f1_pred_from_Rx.eval()
        loss = 0
        with torch.no_grad():
            for X,Rx,Y,weight, setidx in Loader_val:
                X = torch.squeeze(X,dim=0)
                X = X.to(device)
                Y = torch.squeeze(Y,dim=0)
                Y = Y.to(device)
                Rx_est1,_ = net(X)
                Rx_est2,_ = net2(X)
                Rx_est1 = Rx_est1.detach()
                Rx_est2 = Rx_est2.detach()
                Rx = torch.cat((Rx_est1,Rx_est2),dim=1)
                Yhat = f1_pred_from_Rx(Rx)
                lossi = MSEloss(Yhat, Y[:,0].long())
                loss = loss + lossi
        loss = loss / len(Loader_val)
        loss_iepoch = loss
        if epoch %10==0:
            print("Val" +'Epoch {}: predict_MSE: {:.4f}'.format(epoch, loss))
        ###########################################

        
        if loss_iepoch < loss_best:
            print("Update Best Model In Epoch:{:4f} with val loss PRED: {:4f}".format(epoch,loss_iepoch))
            loss_best = loss_iepoch
            torch.save(f1_pred_from_Rx.state_dict(), os.path.join(args.save, "dCORpred_idata_" + The_DATA_MARK  + "Mloop_"  + str(1) + '_f1_Best'  + '_net.pt'))


file_net_dict = os.path.join(args.save, "dCORpred_idata_" + The_DATA_MARK  + "Mloop_"  + str(1) + '_f1_Best'  + '_net.pt')
f1_pred_from_Rx = pred_from_Rx(args.latent_dim*2,outdim =  2)
f1_pred_from_Rx.load_state_dict(torch.load(file_net_dict))



import pickle
x_test = torch.Tensor(XTtest)
with torch.no_grad():
    Rx_pred1,_ = net(x_test)
    Rx_pred2,_ = net2(x_test)
    Rx_pred = torch.cat((Rx_pred1,Rx_pred2),dim=1)
    y_pred = f1_pred_from_Rx(Rx_pred)





out = F.log_softmax(y_pred, dim=0)
y_indpred = torch.max(out ,1)[1]
y_indpred = torch.max(y_pred ,1)[1]




yy = y_indpred.int().numpy()
YTtest1 = YTtest[:,0].astype("int")


AAA = (yy==YTtest1).astype("int")


mse_pred = np.mean(AAA).copy()

print("The  mse {:4f}".format(mse_pred))

res = mse_pred

import pickle
with open(os.path.join(args.save_pickle,  "Result_" + The_DATA_MARK  + "_Mloop_" +  str(idx_data) + '_net_Best' + '_result.pickle'), 'wb') as handle:
    pickle.dump(
        res
        , handle, protocol=pickle.HIGHEST_PROTOCOL)
