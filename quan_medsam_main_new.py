import argparse
import copy
import os

import torch
import torch.nn as nn

from datautils import *
from modelutils import *
from quant import *
from comq_new import *
# from datautils import NpyDataset_tiny
from glob import glob
from os.path import join, exists, isfile, isdir, basename
import random
import cv2




parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default="tiny_vit", choices=['tiny_vit', 'vit_b'])
parser.add_argument('--dataset', type=str, default='imagenet')
parser.add_argument(
    '--compress', type=str, default='quant' ,choices=['quant']
)
parser.add_argument('--load', type=str, default='')
parser.add_argument('--datapath', type=str, default='/network/rit/lab/yinlab/azzhang/GPFQ/data/ILSVRC2012')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--save', type=bool, default=True)

parser.add_argument('--nsamples', type=int, default=1024)
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--workers', type=int, default=1)
parser.add_argument('--nrounds', type=int, default=1)
parser.add_argument('--noaug', action='store_true')

parser.add_argument('--wbits', type=int, default=8)
parser.add_argument('--abits', type=int, default=32)


args = parser.parse_args()

# get some tool functions
get_model, test, run = get_functions(args.model)

aquant = args.compress == 'quant' and args.abits < 32
wquant = args.compress == 'quant' and args.wbits < 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

lite_medsam_model, modelp= get_model() # will quantize model
modeld = copy.deepcopy(lite_medsam_model.image_encoder).to(device) # float model
modeld.eval()
# modelp.eval()
# modelp = copy.deepcopy(lite_medsam_model.image_encoder)

# load
# if args.compress == 'quant' and args.load:
#     modelp.load_state_dict(torch.load(args.load))

# Activation quantization. If you only use weight quantization, you can skip this step. 
if aquant:
    add_actquant(modelp)

# modeld = get_model() # float model

# get the dict of layers and delete the embedding and head layer.
layersp = find_layers(modelp)
layersd = find_layers(modeld)
print(layersp)
# patch_embed = 'patch_embed.proj'
# tiny vit
patch_embed = 'patch_embed.seq.0.c'
neck = 'neck.2'
# vit_b
# patch_embed = 'patch_embed.proj'
# neck = 'neck.2'
# head = 'head'
del layersp[patch_embed]
del layersp[neck]
del layersd[patch_embed]
del layersd[neck]
print(layersp)

# prepare data
# torch.save(modelp.state_dict(), './quan_ckpt/Ablation_Study/medsam_tinyvit_32bit.pth')

data_list = [
        '/data/hp/medsam_auto/train_data/CT/AbdomenCT1K',
        # '/data/hp/medsam_auto/train_data/Endoscopy/split_data/Kvasir_SEG',
        # '/data/hp/medsam_auto/train_data/MR/BraTS_FLAIR',
        # '/data/hp/medsam_auto/train_data/XRay/Chest-Xray-Masks-and-Labels'
    ]
train_dataset = NpyDataset_tiny(data_root_list=data_list, data_aug=False)
quantize_set = random_subset(train_dataset, nsamples=args.nsamples, seed=args.seed)
quantize_loader = DataLoader(quantize_set, batch_size=args.batchsize, shuffle=True, num_workers=8,
                          pin_memory=False)

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
            # ab study perchannel True(ours) False
            args.wbits, perchannel=False, sym=False, mse=True
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
        for j, batch in enumerate(quantize_loader):
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
        run(modeld, next(iter(quantize_loader)))
    for h in handles:
        h.remove()

if args.save:
    from os import makedirs
    makedirs('./quan_ckpt/perlayer', exist_ok=True)
    torch.save(modelp.state_dict(), './quan_ckpt/perlayer/medsam_tinyvit_8bit.pth')

if args.wbits == 100:
    torch.cuda.empty_cache()
    from torch import optim
    from tqdm import tqdm
    train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8,)
    freeze_layers_tinyvit(modelp)
    n_epochs = 1
    lr = 1 * 1e-4
    weight_decay = 1e-6
    pg = [p for p in modelp.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=lr, weight_decay=weight_decay)

    modelp.train()
    modeld.eval()
    modelp.to(train_device)
    modeld.to(train_device)
    l2loss = nn.MSELoss()
    for epoch in range(1, n_epochs + 1):
        batch_losses = []
        for step, batch in enumerate(tqdm(train_loader)):
            image = batch["image"]
            image = image.to(train_device)
            with torch.no_grad():
                image_embedding = modeld(image)  # (b,256,64,64)
            quantized_embeding = modelp(image)
            loss = l2loss(quantized_embeding, image_embedding)
            optimizer.step()
            optimizer.zero_grad()
            batch_losses.append(loss.item())
    modelp.eval()
    from os import makedirs
    makedirs('./quan_ckpt/batchtunning', exist_ok=True)
    torch.save(modelp.state_dict(), './quan_ckpt/batchtunning/medsam_tinyvit_4bit.pth')


