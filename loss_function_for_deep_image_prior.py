import torch
import torch.nn as nn
import numpy as np

class MSELossUncertainty(nn.Module):
    '''
        Given noisy measurements g^\delta=(g^\delta_1,...,g^\delta_n)^T, the loss specified in the README file is calculated in self.forward().
    '''
    def __init__(self, g_noisy, c):
        super(MSELossUncertainty, self).__init__()
        self.g_noisy = g_noisy
        self.c       = c.view(1,1, g_noisy.size()[-1])
        self.MSE     = nn.MSELoss()
    
    def forward(self, fx):
        '''
            fx is the vector of A^\eta_i applied to the DIP \varphi_\theta(z), see again the README file.
        '''
        z1 = torch.abs(fx - self.g_noisy)**2
        return self.MSE(z1, self.c)
    