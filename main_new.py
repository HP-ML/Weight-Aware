import argparse
import copy
import os

import torch
import torch.nn as nn

from datautils import *
from modelutils import *
from quant import *
from comq_new import *


parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str)
parser.add_argument('--dataset', type=str, default='imagenet')
parser.add_argument(
    '--compress', type=str, choices=['quant']
)
parser.add_argument('--load', type=str, default='')
parser.add_argument('--datapath', type=str, default='/network/rit/lab/yinlab/azzhang/GPFQ/data/ILSVRC2012')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--save', type=str, default='')

parser.add_argument('--nsamples', type=int, default=1024)
parser.add_argument('--batchsize', type=int, default=-1)
parser.add_argument('--workers', type=int, default=1)
parser.add_argument('--nrounds', type=int, default=1)
parser.add_argument('--noaug', action='store_true')

parser.add_argument('--wbits', type=int, default=4)
parser.add_argument('--abits', type=int, default=32)


args = parser.parse_args()

# get some tool functions
get_model, test, run = get_functions(args.model)

aquant = args.compress == 'quant' and args.abits < 32
wquant = args.compress == 'quant' and args.wbits < 32

modelp = get_model() # will quantize model

if args.compress == 'quant' and args.load:
    modelp.load_state_dict(torch.load(args.load))

# Activation quantization. If you only use weight quantization, you can skip this step. 
if aquant:
    add_actquant(modelp)

modeld = get_model() # float model

# get the dict of layers and delete the embedding and head layer.
layersp = find_layers(modelp)
layersd = find_layers(modeld)
patch_embed = 'patch_embed.proj'
head = 'head'
del layersp[patch_embed]
del layersp[head]
del layersd[patch_embed]
del layersd[head]
print(layersp)

# prepare data
dataloader, testloader = get_loaders(
    args.dataset, modelp, path=args.datapath, 
    batchsize=args.batchsize, workers=args.workers,
    nsamples=args.nsamples, seed=args.seed,
    noaug=args.noaug
)


comq = {}
for name in layersp:
    layer = layersp[name]
    if isinstance(layer, ActQuantWrapper):
        layer = layer.module
    comq[name] = COMQ(layer)
    if aquant:
        layersp[name].quantizer.configure(
            args.abits, sym=args.asym, mse=not args.aminmax
        )
    if wquant:
        comq[name].quantizer = Quantizer()
        comq[name].quantizer.configure(
            
            args.wbits, perchannel=True, sym=False, mse=True
        )

if not (args.compress == 'quant' and not wquant):
    cache = {}
    def add_batch(name):
        def tmp(layer, inp, out):
            comq[name].add_batch(inp[0].data, out.data)
        return tmp
    handles = []
    for name in comq:
        handles.append(layersd[name].register_forward_hook(add_batch(name)))
    for i in range(args.nrounds):
        for j, batch in enumerate(dataloader):
            print(i, j)
            with torch.no_grad():
                run(modeld, batch)
    for h in handles:
        h.remove()
    for name in comq:
        print(name)
        if args.compress == 'quant':
            print('Quantizing ...')
            comq[name].quantize()
        
        comq[name].free()

# Activation quantization.
if aquant:
    print('Quantizing activations ...')
    def init_actquant(name):
        def tmp(layer, inp, out):
            layersp[name].quantizer.find_params(inp[0].data)
        return tmp
    handles = []
    for name in layersd:
        handles.append(layersd[name].register_forward_hook(init_actquant(name)))
    with torch.no_grad():
        run(modeld, next(iter(dataloader)))
    for h in handles:
        h.remove()

if args.save:
    torch.save(modelp.state_dict(), args.save)

test(modelp, testloader)
