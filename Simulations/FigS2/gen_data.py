import numpy as np
from scipy.linalg import sqrtm
from scipy.special import expit
import numpy as np


def generate_orthogonal_vectors(d):
    """
    生成7个维度为d的相互正交向量，使用傅里叶基
    
    参数:
        d: 向量的维度，需要满足d >= 7
    
    返回:
        vectors: 7个相互正交的向量组成的数组，形状为(7, d)
    """
    if d < 7:
        raise ValueError("维度d必须大于等于7")
    
    # 初始化向量数组
    vectors = np.zeros((7, d))
    
    # 定义向量索引
    t = np.arange(1, d+1)
    
    # 常数向量 e₀
    vectors[0] = np.ones(d) / np.sqrt(d)
    
    # j=1 的余弦和正弦向量 e₁, e₂
    vectors[1] = np.cos(2 * np.pi * 1 * t / d) / np.sqrt(d/2)
    vectors[2] = np.sin(2 * np.pi * 1 * t / d) / np.sqrt(d/2)
    
    # j=2 的余弦和正弦向量 e₃, e₄
    vectors[3] = np.cos(2 * np.pi * 2 * t / d) / np.sqrt(d/2)
    vectors[4] = np.sin(2 * np.pi * 2 * t / d) / np.sqrt(d/2)
    
    # j=3 的余弦和正弦向量 e₅, e₆
    vectors[5] = np.cos(2 * np.pi * 3 * t / d) / np.sqrt(d/2)
    vectors[6] = np.sin(2 * np.pi * 3 * t / d) / np.sqrt(d/2)
    # for i in range(7):
    #     print(f"e_{i} 的范数: {np.linalg.norm(vectors[i])}")
    #     print(f'e_{i} 的均值: {np.mean(vectors[i])}')
    return vectors

def verify_orthogonality(vectors):
    n = vectors.shape[0]
    dot_products = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            dot_products[i, j] = np.dot(vectors[i], vectors[j])
    
    # print("向量内积矩阵:")
    # print(np.round(dot_products, 10))  # 四舍五入到10位小数，处理浮点误差
    
    # 检查是否为单位矩阵(对角线为1，其他为0)
    is_orthogonal = np.allclose(dot_products, np.eye(n), atol=1e-10)
    # print(f"向量是否正交: {is_orthogonal}")




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
        vectors = generate_orthogonal_vectors(P)
        verify_orthogonality(vectors)
        print("vectors.shape",vectors.shape)
        # print("vectors",vectors)
        facotr_list = []
        for i in range(len(vectors)):
            facotr_list.append(np.dot(X,vectors[i]))

        # for i in range(len(facotr_list)):
        #     print(f"factor_{i} 的均值: {np.mean(facotr_list[i])}")
        #     print(f"factor_{i} 的标准差: {np.std(facotr_list[i])}")
        ff0 = 1*(facotr_list[0]-0.9)**2
        ff1 = 1*(facotr_list[1]-0.5)**2 * (- 1*facotr_list[2]*facotr_list[1] )
        ff2 = np.sin(-3.14*facotr_list[2]*facotr_list[3]/5) + 1
        ff3 = facotr_list[3]*(np.abs(facotr_list[4]) + 1)**2
        ff4 = np.sin(np.pi *0.5*facotr_list[5]) + 1
        ff5 = 2*np.sin(np.pi * facotr_list[6]) / (2  -  np.sin(np.pi * facotr_list[6])) 
        # print(np.mean(ff0), np.mean(ff1), np.mean(ff2), np.mean(ff3), np.mean(ff4), np.mean(ff5))
        # print(np.std(ff0), np.std(ff1), np.std(ff2), np.std(ff3), np.std(ff4), np.std(ff5))


        
        # ff0 = 1*(X[:,0]-0.9)**2 
        # ff1 = 1*(X[:,1]-0.5)**2 * (- 1*X[:,2]*X[:,1] )
        # ff2 = np.sin(-3.14*X[:,2]*X[:,3]/5) + 1
        # ff3 = X[:,3]*(np.abs(X[:,4]) + 1)**2
        # ff4 = np.sin(np.pi *0.5*X[:,5]) + 1
        # ff5 = 2*np.sin(np.pi * X[:,6]) / (2  -  np.sin(np.pi * X[:,6]))




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
        # Prob_y = 1/(1+ (np.exp(y0))**(-1)) #original
        # Prob_y = 1/(1+ np.exp(-y0))  #xueyu
        Prob_y = expit(y0)  # Xueyu
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

    # if s==4 or s==5 or s==6:
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
        self.nEpochs = 100
        self.nEpochs_pred = 100
        self.batch_size = 64
        self.lr_R = 1*1e-3
        self.lr_D = 1*1e-3
        self.lr_pred = 1*1e-3
        # self.cuda = True
        self.cuda = False
        self.lr_step = 100
        self.decayRate =0.8
        self.Nnumber = 1