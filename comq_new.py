import math
import time

import torch
import torch.nn as nn
from sympy import group

from quant import *
# from MagR import *


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEBUG = False 


class COMQ:

    def __init__(self, layer, rel_damp=0):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            # self.rows = self.layer.in_channels
            # self.columns = self.layer.out_channels
            W = W.flatten(1)  # shape out_feature*in_channels/groups*3*3  out_feature*(in_channels/groups*9)
            # self.rows = W.shape[0] * self.layer.groups
            # self.columns = W.shape[1] * self.layer.groups

            self.rows = W.shape[0]
            self.columns = W.shape[1]
        else:
            self.rows = W.shape[0]
            self.columns = W.shape[1]
        
        self.XtX = torch.zeros((self.columns, self.columns), device=self.dev)
        

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            elif len(inp.shape) == 4:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            # dedpwise
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            if self.layer.groups > 1:  # need confirm
                inp = unfold(inp) # b, hidden dim *kernel , feature size*
                inp = inp.reshape((inp.shape[0], self.columns, self.rows, inp.shape[-1] ))
                inp = inp.permute([1, 0, 2, 3])
                inp = inp.flatten(1)
            else:
                inp = unfold(inp)
                inp = inp.permute([1, 0, 2])
                inp = inp.flatten(1)
        
        self.X = inp.t().float()
        self.XtX += inp.float().matmul(inp.float().t()) 
        

    def prepare(self, columnslast=False):

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            # if self.layer.groups > 1:
            #
            W = W.flatten(1)

        return W




    def quantize(self, groupsize=-1):
        W = self.prepare()
        Q_CD = W.clone()
        W_orig = W.clone()
        self.quantizer.find_params(W, weight=True)

        print('comq')
        tick = time.time()

        self.XtX = self.XtX.to(self.dev)
        diag_Sigma = torch.diagonal(self.XtX, 0)
        diag_Sigma += 0.75*torch.mean(torch.diagonal(self.XtX))
        norm_Sigma = torch.div(self.XtX, diag_Sigma+0.1) 
            
        P = torch.matmul(W_orig, norm_Sigma)
            
        norm_Sigma.fill_diagonal_(0)
        norm_Sigma = norm_Sigma.t()

        for _ in range(10):
                
            delta_Q = Q_CD.clone().t()
                
            P_hat = torch.matmul(norm_Sigma, delta_Q).t()
                
            for j in range(self.columns):
                    
                u = P[:, j] - P_hat[:, j]
                    
                if j > 0:
                    
                    u += torch.matmul(norm_Sigma[j, :j], delta_Q[:j, :])

                if groupsize != -1:
                        
                    if not static_groups:
                            
                        if j % groupsize == 0:
                            
                            self.quantizer.find_params(Q_CD[:, j:(j + groupsize)], weight=True)

                    else:
                            
                        idx = j
                            
                        if actorder:
                            idx = perm[idx]
                            
                        self.quantizer = groups[idx // groupsize]

                u = quantize(u.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq).flatten()
                
                Q_CD[:, j] = u
                    
                delta_Q[j, :] -= u
        print('time %.2f' % (time.time() - tick))
        quantization_error_weight = torch.norm(W_orig-Q_CD, p='fro')
        # quantization_error = torch.norm(self.X @ W.T - self.X @ Q_CD.T, p='fro')
        print(f'The weight quantization error is {quantization_error_weight}')
        # print(f'The weight quantization error is {quantization_error}')
        # print(torch.norm(self.X @ W.T - self.X @ Q_CD.T, p='fro'))
        # print(self.X (W_orig-Q_CD))

        self.layer.weight.data = Q_CD.reshape(self.layer.weight.shape)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2) / 128)


    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
