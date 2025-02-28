import math
import time

import torch
import torch.nn as nn

from quant import *
# from MagR import *
# from test1 import *

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEBUG = False 


class COMQ:

    def __init__(self, layer, rel_damp=0):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
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
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            if self.layer.groups > 1:  # need confirm
                inp = unfold(inp)  # b, hidden dim *kernel , feature size*
                inp = inp.reshape((inp.shape[0], self.columns, self.rows, inp.shape[-1]))
                inp = inp.permute([1, 0, 2, 3])
                inp = inp.flatten(1)
            else:
                inp = unfold(inp)
                inp = inp.permute([1, 0, 2])
                inp = inp.flatten(1)
            # inp = unfold(inp)
            # inp = inp.permute([1, 0, 2])
            # inp = inp.flatten(1)
        
        self.X = inp.t().float()
        self.XtX += inp.float().matmul(inp.float().t()) 
        

    def prepare(self, columnslast=False):

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        return W


    def quantize(self, groupsize=-1):
        W = self.prepare()
        
        self.quantizer.find_params(W, weight=True)

        Q = torch.zeros_like(W).to(self.dev)

        if groupsize != -1:
            
            
            for j in range(self.columns):
                
                w = W[:, j]

                if groupsize != -1:
                    
                    if not static_groups:
                        
                        if j % groupsize == 0:
                            self.quantizer.find_params(W[:, j:(j + groupsize)], weight=True)

                    else:  
                        idx = j
                        
                        if actorder:
                            idx = perm[idx]
                        
                        self.quantizer = groups[idx // groupsize]

                q = quantize(w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq).flatten()
                
                Q[:, j] = q


            
        else:

            Q = quantize(W, self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq)

        
        self.layer.weight.data = Q.reshape(self.layer.weight.shape)
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
