import numpy as np
from scipy.linalg import sqrtm
from scipy.special import expit
import numpy as np


def generate_orthogonal_vectors(d):

    if d < 7:
        raise ValueError("维度d必须大于等于7")
    

    vectors = np.zeros((7, d))
    

    t = np.arange(1, d+1)
    

    vectors[0] = np.ones(d) / np.sqrt(d)
    

    vectors[1] = np.cos(2 * np.pi * 1 * t / d) / np.sqrt(d/2)
    vectors[2] = np.sin(2 * np.pi * 1 * t / d) / np.sqrt(d/2)
    

    vectors[3] = np.cos(2 * np.pi * 2 * t / d) / np.sqrt(d/2)
    vectors[4] = np.sin(2 * np.pi * 2 * t / d) / np.sqrt(d/2)
    

    vectors[5] = np.cos(2 * np.pi * 3 * t / d) / np.sqrt(d/2)
    vectors[6] = np.sin(2 * np.pi * 3 * t / d) / np.sqrt(d/2)

    return vectors

def verify_orthogonality(vectors):
    n = vectors.shape[0]
    dot_products = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            dot_products[i, j] = np.dot(vectors[i], vectors[j])

    is_orthogonal = np.allclose(dot_products, np.eye(n), atol=1e-10)




def standardization(ddata):
    mu = np.mean(ddata, axis=0)
    sigma = np.std(ddata, axis=0)
    return (ddata - mu) / sigma


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
        vectors = generate_orthogonal_vectors(P)
        verify_orthogonality(vectors)
        print("vectors.shape",vectors.shape)

        facotr_list = []
        for i in range(len(vectors)):
            facotr_list.append(np.dot(X,vectors[i]))


        ff0 = 1*(facotr_list[0]-0.9)**2
        ff1 = 1*(facotr_list[1]-0.5)**2 * (- 1*facotr_list[2]*facotr_list[1] )
        ff2 = np.sin(-3.14*facotr_list[2]*facotr_list[3]/5) + 1
        ff3 = facotr_list[3]*(np.abs(facotr_list[4]) + 1)**2
        ff4 = np.sin(np.pi *0.5*facotr_list[5]) + 1
        ff5 = 2*np.sin(np.pi * facotr_list[6]) / (2  -  np.sin(np.pi * facotr_list[6])) 




    f0 = ff0
    f1 = ff1
    f2 = ff2
    f3 = ff3
    f4 = ff4
    f5 = ff5



    if s == 0:
        
        mu0 =  2*f0 + 1*f1 + 1*f2 + 1*f3

        y0 = mu0 - mu0.mean()
        y0 = y0.reshape(-1,1)
        y0 = y0.astype(np.float32)
        indicator_y0 = np.ones_like(y0)
        Prob_y = expit(y0)  
        for i in range(len(Prob_y)):
            indicator_y0[i] = np.random.binomial(1, Prob_y[i], size=1)
        y = indicator_y0

 
    sigma = 0.5
    if s==1 or s==2:
        mu1  = 3*f0 + 1*f1 - 1*f2 + s*f4
        eps1 = np.random.randn(n)
        y1 = mu1 + sigma*eps1
        y1 = y1.reshape(-1,1).astype(np.float32)
        y = y1

    if s==4 or s==3:
        ss = s - 2
        mu1  = 2*f0 - 1.5*f1 + 1*f2  + ss*f5
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
        self.nEpochs = 200
        self.nEpochs_pred = 200
        self.batch_size = 64
        self.lr_R = 1*1e-3
        self.lr_D = 1*1e-3
        self.lr_pred = 1*1e-3

        self.cuda = False
        self.lr_step = 100
        self.decayRate =0.8
        self.Nnumber = 1