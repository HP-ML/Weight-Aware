import argparse
import copy
import os

import torch
import torch.nn as nn

from datautils import *
from modelutils import *
from quant import *
# from comq_new import *
# from comq_new_permuation import *   #
from rtn import *
from glob import glob
from os.path import join, exists, isfile, isdir, basename
import random
import cv2




parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default="tiny_vit")
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

parser.add_argument('--wbits', type=int, default=2)
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
patch_embed = 'patch_embed.seq.0.c'
neck = 'neck.2'
# head = 'head'
del layersp[patch_embed]
del layersp[neck]
del layersd[patch_embed]
del layersd[neck]
print(layersp)

# prepare data

def resize_longest_side(image, target_length):
    """
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    long_side_length = target_length
    oldh, oldw = image.shape[0], image.shape[1]
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    target_size = (neww, newh)

    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def pad_image(image, target_size):
    """
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    # Pad
    h, w = image.shape[0], image.shape[1]
    padh = target_size - h
    padw = target_size - w
    if len(image.shape) == 3:  ## Pad image
        image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
    else:  ## Pad gt mask
        image_padded = np.pad(image, ((0, padh), (0, padw)))

    return image_padded
class NpyDataset(Dataset):

    def __init__(self, data_root_list, image_size=256, bbox_shift=5, data_aug=False):
        self.data_root_list = data_root_list
        self.gt_path_files = []
        self.img_path_files = []

        # Collect files from all data roots
        for data_root in data_root_list:
            gt_path = join(data_root, 'gts')
            img_path = join(data_root, 'imgs')
            gt_files = sorted(glob(join(gt_path, '*.npy'), recursive=True))
            gt_files = [
                file for file in gt_files
                if isfile(join(img_path, basename(file)))
            ]

            self.gt_path_files.extend(gt_files)
            self.img_path_files.extend([join(img_path, basename(file)) for file in gt_files])

        self.image_size = image_size
        self.target_length = image_size
        self.bbox_shift = bbox_shift
        self.data_aug = data_aug

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = basename(self.gt_path_files[index])
        assert img_name == basename(self.gt_path_files[index]), 'img gt name error: ' + self.gt_path_files[index]

        img_3c = np.load(self.img_path_files[index], 'r', allow_pickle=True)  # (H, W, 3)
        img_resize = resize_longest_side(img_3c, 256)
        # Resizing
        img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8,
                                                               a_max=None)  # normalize to [0, 1], (H, W, 3)
        img_padded = pad_image(img_resize, 256)  # (256, 256, 3)
        # convert the shape to (3, H, W)
        img_padded = np.transpose(img_padded, (2, 0, 1))  # (3, 256, 256)
        assert np.max(img_padded) <= 1.0 and np.min(img_padded) >= 0.0, 'image should be normalized to [0, 1]'

        gt = np.load(self.gt_path_files[index], 'r', allow_pickle=True)  # multiple labels [0, 1, 4, 5...], (256,256)
        gt = cv2.resize(
            gt,
            (img_resize.shape[1], img_resize.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)
        gt = pad_image(gt, 256)  # (256, 256)

        label_ids = np.unique(gt)[1:]
        try:
            gt2D = np.uint8(gt == random.choice(label_ids.tolist()))  # only one label, (256, 256)
        except:
            print(img_name, 'label_ids.tolist()', label_ids.tolist())
            gt2D = np.uint8(gt == np.max(gt))  # only one label, (256, 256)

        # add data augmentation: random fliplr and random flipud
        if self.data_aug:
            if random.random() > 0.5:
                img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-1))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-1))
            if random.random() > 0.5:
                img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-2))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-2))

        gt2D = np.uint8(gt2D > 0)
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])

        return {
            "image": torch.tensor(img_padded).float(),
            "gt2D": torch.tensor(gt2D[None, :, :]).long(),
            "bboxes": torch.tensor(bboxes[None, None, ...]).float(),  # (B, 1, 4)
            "image_name": img_name,
            "new_size": torch.tensor(np.array([img_resize.shape[0], img_resize.shape[1]])).long(),
            "original_size": torch.tensor(np.array([img_3c.shape[0], img_3c.shape[1]])).long()
        }

data_list = [
        '/data/hp/medsam_auto/train_data/CT/AbdomenCT1K',
        # '/data/hp/medsam_auto/train_data/Endoscopy/split_data/Kvasir_SEG',
        # '/data/hp/medsam_auto/train_data/MR/BraTS_FLAIR',
        # '/data/hp/medsam_auto/train_data/XRay/Chest-Xray-Masks-and-Labels'
    ]
train_dataset = NpyDataset(data_root_list=data_list, data_aug=False)
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
    makedirs('./quan_ckpt/rtn', exist_ok=True)
    torch.save(modelp.state_dict(), './quan_ckpt/rtn/medsam_tinyvit_2bit.pth')

if args.wbits == 200:
    torch.cuda.empty_cache()
    from torch import optim
    from tqdm import tqdm
    train_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
    makedirs('./quan_ckpt/batchtunning/permuation', exist_ok=True)
    torch.save(modelp.state_dict(), './quan_ckpt/batchtunning/permuation/medsam_tinyvit_2bit.pth')


