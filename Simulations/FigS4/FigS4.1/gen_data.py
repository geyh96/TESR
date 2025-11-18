import numpy as np
from scipy.linalg import sqrtm
import numpy as np


def standardization(ddata):
    mu = np.mean(ddata, axis=0)
    sigma = np.std(ddata, axis=0)
    return (ddata - mu) / sigma

#idx_domain = 0 is the target dataset
def Get_data_0101(n, P, s=1):
    Cov_z = np.eye(P)
    for k in range(P):
            for l in range(P):
                    Cov_z[k,l] = 0.2**(np.abs(k-l))
    sqrtCov_z = sqrtm(Cov_z)
    Z0 = np.random.randn(n,P)
    Z = Z0 @ sqrtCov_z
    X = Z

    if 0:
        ff0 = X[:,0]
        ff1 = 2*X[:,1] + 1
        ff2 = 2*X[:,2] - 1
        ff3 = 0.1 * np.sin(np.pi*X[:,3]) + 0.2 * np.cos(np.pi*X[:,3]) + 0.3 * np.sin(np.pi*X[:,3])**2 + 0.4 * np.cos(np.pi*X[:,3])**2 + 0.5 * np.sin(np.pi*X[:,3])**3 
        ff4 = np.sin(np.pi * X[:,4]) / (2  -  np.sin(np.pi * X[:,4]))
        ff5 = X[:,4]*(np.abs(X[:,5]) + 1)**2
    if 1:
        ff0 = 1*(X[:,0]-0.9)**2 
        ff1 = 1*(X[:,1]-0.5)**2 * (- 1*X[:,2]*X[:,1] )
        ff2 = np.sin(-3.14*X[:,2]*X[:,3]/5) + 1
        ff3 = X[:,3]*(np.abs(X[:,4]) + 1)**2
        ff4 = np.sin(np.pi *0.5*X[:,5]) + 1
        ff5 = 2*np.sin(np.pi * X[:,6]) / (2  -  np.sin(np.pi * X[:,6]))




    f0 = ff0
    f1 = ff1
    f2 = ff2
    f3 = ff3
    f4 = ff4
    f5 = ff5



    if s == 0:
        # mu0 =  2*f0 + 1*f1 + 1*f2  + 1*f5
        mu0 =  2*f0 + 1*f1 + 1*f2 + 1*f3
        # eps = np.random.randn(n)
        y0 = mu0 - mu0.mean()
        y0 = y0.reshape(-1,1)
        y0 = y0.astype(np.float32)
        indicator_y0 = np.ones_like(y0)
        Prob_y = 1/(1+ (np.exp(y0))**(-1))
        for i in range(len(Prob_y)):
            indicator_y0[i] = np.random.binomial(1, Prob_y[i], size=1)
        y = indicator_y0

 
    sigma = 0.5
    if s==1 or s==2:
        mu1  = 3*f0 + 1*f1 + 1*f2 + s*f4
        eps1 = np.random.randn(n)
        y1 = mu1 + sigma*eps1
        y1 = y1.reshape(-1,1).astype(np.float32)
        y = y1

    # if s==4 or s==5 or s==6:
    if s==4 or s==3:
        ss = s - 2
        mu1  = 2*f0 + 1.5*f1 + 1*f2  + ss*f5
        eps1 = np.random.randn(n)
        y1 = mu1 + sigma*eps1
        y1 = y1.reshape(-1,1).astype(np.float32)
        y = y1

    X = X.astype(np.float32)

    dataset = {"X":X,
            "y":y,
            }
    return dataset




class class_args:
    def __init__(self):
        super().__init__()
        self.seed = 20240604
        self.m = 4
        self.NTest = 10000
        self.list_lambda_predloss = [0]
        self.list_lambda_Eloss = [0.1]
        self.list_lambda_Eloss_Source = [0.1]
        self.list_lambda_priorloss = [0.1]
        self.list_lambda_Invarloss_Source = [0.1]
        self.list_lambda_IRMloss_Source = [0.1]
        self.latent_dim = 64
        self.save = "./model/"
        self.save_pickle = "./result/"
        self.nEpochs = 300
        self.nEpochs_pred = 300
        self.batch_size = 64
        self.lr_R = 1*1e-3
        self.lr_D = 1*1e-3
        self.lr_pred = 1*1e-3
        # self.cuda = True
        self.cuda = False
        self.lr_step = 100
        self.decayRate =0.8
        self.Nnumber = 1