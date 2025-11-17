import torch
import torch.nn as nn


class Loss_DC(nn.Module):
    def __init__(self):
        super(Loss_DC, self).__init__()
    def Distance_Correlation(self, Z, Y):
        matrix_a = torch.sum(torch.square(Z.unsqueeze(1) - Z.unsqueeze(0)), dim = -1) + 1e-12
        matrix_b = torch.sum(torch.square(Y.unsqueeze(1) - Y.unsqueeze(0)), dim = -1) + 1e-12
        matrix_A_i_meanj = torch.mean(matrix_a,dim=0,keepdims=True)
        matrix_A_meani_j = torch.mean(matrix_a,dim=1,keepdims=True)
        matrix_A_meanij = torch.mean(matrix_a)
        matrixA = matrix_a - matrix_A_i_meanj - matrix_A_meani_j + matrix_A_meanij
        

        matrix_B_i_meanj = torch.mean(matrix_b,dim=0,keepdims=True)
        matrix_B_meani_j = torch.mean(matrix_b,dim=1,keepdims=True)
        matrix_B_meanij = torch.mean(matrix_b)
        matrixB = matrix_b - matrix_B_i_meanj - matrix_B_meani_j + matrix_B_meanij

        
        matrix_AA = torch.mul(matrixA, matrixA)
        matrix_BB = torch.mul(matrixB, matrixB)
        matrix_AB = torch.mul(matrixB, matrixA)
        Gamma_XY = torch.mean(matrix_AB)#
        Gamma_XX = torch.mean(matrix_AA)#
        Gamma_YY = torch.mean(matrix_BB)#


        correlation_r = Gamma_XY/torch.sqrt(Gamma_XX * Gamma_YY + 1e-9)
        return correlation_r

    def forward(self, Z, Y):
        dc_loss = self.Distance_Correlation(Z, Y)
        return dc_loss






class Loss_Energy(nn.Module):
    def __init__(self):
        super(Loss_Energy, self).__init__()
    def Distance_Correlation(self, Z, D):
        
        matrix_z = torch.sum(torch.square(Z.unsqueeze(0) - Z.unsqueeze(1)), dim = -1) + 1e-12
        matrix_d = torch.sum(torch.square(D.unsqueeze(0) - D.unsqueeze(1)), dim = -1) + 1e-12
        matrix_zd = torch.sum(torch.square(Z.unsqueeze(1) - D.unsqueeze(0)), dim = -1) + 1e-12
        matrix_Z = torch.mean(matrix_z)
        matrix_D = torch.mean(matrix_d)
        matrix_ZD = torch.mean(matrix_zd)
        correlation_r = (2*matrix_ZD - matrix_D - matrix_Z+ 1e-12)/(2*matrix_ZD + 1e-12) 
        return correlation_r
    def forward(self, Z, Y):
        dc_loss = self.Distance_Correlation(Z, Y)
        return dc_loss
