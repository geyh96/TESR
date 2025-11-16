import numpy as np
from scipy.linalg import sqrtm
import numpy as np
import scipy
n=1000
P=10
# rotation = 30
# power = 1
# s = 1
print(np.cos(90/180*np.pi))
print(np.cos(90/180*np.pi))
def standardization(ddata):
    mu = np.mean(ddata, axis=0)
    sigma = np.std(ddata, axis=0)
    return (ddata - mu) / sigma

#idx_domain = 0 is the target dataset
power = 1
rotation = 0
Cov_z = np.eye(P)
for k in range(P):
        for l in range(P):
                Cov_z[k,l] = 0**(np.abs(k-l))
                # Cov_z[k,l] = 0.2**(np.abs(k-l))
sqrtCov_z = sqrtm(Cov_z)
Z0 = np.random.randn(n,P)
Z = Z0 @ sqrtCov_z
X = Z
rotation1 = rotation/180*np.pi
rotation_matrix = np.array([[np.cos(rotation1), -np.sin(rotation1)],[np.sin(rotation1), np.cos(rotation1)]])
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
    ff4 = np.sin(-1.7*X[:,3]*X[:,5]) + 1
    ff5 = 2*np.sin(np.pi * X[:,5]) / (2  -  np.sin(np.pi * X[:,5]))



f0 = ff0
f1 = ff1
f2 = ff2
f3 = ff3
f4 = ff4
f5 = ff5


sigma = 0.5
# if s==1 or s==2 or s==3:
s=1
# mu1  = 2*f0 + 1*f1 + 1*f2 + s*f3
# mu1  = 2*paras[0]*f0 + paras[1]*f1 + f2 + 1*f3
# mu1.std()
mu0  = (2)*f0 + (1)*(f1 + f2) + 1*f3
mu1  = (2+power)*f0 + (1+power)*(f1 + f2) + s*f4
mu0.std()
mu1.std()


