import os
import sys
sys.path.append('yolov5')

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from glob import glob
from os.path import join, exists, isfile, isdir, basename
import cv2
import random
from skimage import transform



def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def random_subset(data, nsamples, seed):
    set_seed(seed)
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    return Subset(data, idx[:nsamples])


_IMAGENET_RGB_MEANS = (0.485, 0.456, 0.406)
_IMAGENET_RGB_STDS = (0.229, 0.224, 0.225)

def get_imagenet(model, path, noaug=False):
    # img_size = 224  # standard
    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=_IMAGENET_RGB_MEANS, std=_IMAGENET_RGB_STDS),
    # ])
    # non_rand_resize_scale = 256.0 / 224.0  # standard
    # test_transform = transforms.Compose([
    #     transforms.Resize(round(non_rand_resize_scale * img_size)),
    #     transforms.CenterCrop(img_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=_IMAGENET_RGB_MEANS, std=_IMAGENET_RGB_STDS),
    # ])

    config = resolve_data_config(model.pretrained_cfg)
    train_transform = create_transform(**config, is_training=True)
    test_transform = create_transform(**config)

    train_dir = os.path.join(os.path.expanduser(path), 'ILSVRC2012_img_train')
    test_dir = os.path.join(os.path.expanduser(path), 'val')

    if noaug:
        train_dataset = datasets.ImageFolder(train_dir, test_transform)
    else:
        train_dataset = datasets.ImageFolder(train_dir, train_transform)
    test_dataset = datasets.ImageFolder(test_dir, test_transform)

    return train_dataset, test_dataset

class YOLOv5Wrapper(Dataset):
    def __init__(self, original):
        self.original = original
    def __len__(self):
        return len(self.original)
    def __getitem__(self, idx):
        tmp = list(self.original[idx])
        tmp[0] = tmp[0].float() / 255
        return tmp

def get_coco(path, batchsize):
    from yolov5.utils.datasets import LoadImagesAndLabels
    train_data = LoadImagesAndLabels(
        os.path.join(path, 'images/calib'), batch_size=batchsize
    )
    train_data = YOLOv5Wrapper(train_data)
    train_data.collate_fn = LoadImagesAndLabels.collate_fn
    test_data = LoadImagesAndLabels(
        os.path.join(path, 'images/val2017'), batch_size=batchsize, pad=.5
    )
    test_data = YOLOv5Wrapper(test_data)
    test_data.collate_fn = LoadImagesAndLabels.collate_fn
    return train_data, test_data


DEFAULT_PATHS = {
    'imagenet': [
        '../imagenet'
    ],
    'coco': [
        '../coco'
    ]
}

def get_loaders(
    name, model, path='', batchsize=-1, workers=8, nsamples=1024, seed=0,
    noaug=False
):
    if name == 'squad':
        if batchsize == -1:
            batchsize = 16
        import bertsquad
        set_seed(seed)
        return bertsquad.get_dataloader(batchsize, nsamples), None

    if not path:
        for path in DEFAULT_PATHS[name]:
            if os.path.exists(path):
                break

    if name == 'imagenet':
        if batchsize == -1:
            batchsize = 128
        train_data, test_data = get_imagenet(model, path, noaug=noaug)
        train_data = random_subset(train_data, nsamples, seed)
    if name == 'coco':
        if batchsize == -1:
            batchsize = 16
        train_data, test_data = get_coco(path, batchsize)

    collate_fn = train_data.collate_fn if hasattr(train_data, 'collate_fn') else None
    trainloader = DataLoader(
        train_data, batch_size=batchsize, num_workers=workers, pin_memory=True, shuffle=True, persistent_workers=True,
        collate_fn=collate_fn
    )
    collate_fn = test_data.collate_fn if hasattr(test_data, 'collate_fn') else None
    testloader = DataLoader(
        test_data, batch_size=batchsize, num_workers=workers, pin_memory=True, shuffle=False, persistent_workers=True,
        collate_fn=collate_fn
    )

    return trainloader, testloader

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
class NpyDataset_tiny(Dataset):

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


class NpyDataset_vit_b(Dataset):
    def __init__(self, data_root, bbox_shift=5):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob(join(self.gt_path, "*.npy"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        self.bbox_shift = bbox_shift
        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )
        img_1024 = transform.resize(
            img_1024, (1024, 1024), order=3, preserve_range=True, mode="constant", anti_aliasing=True
        ).astype(np.uint8) # (1024, 1024, 3)

        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )  # normalize to [0, 1], (H, W, 3)
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))

        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )  # multiple labels [0, 1,4,5...], (256,256)
        assert img_name == os.path.basename(self.gt_path_files[index]), (
            "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        )
        gt = transform.resize(
            gt, (1024, 1024), order=0, preserve_range=True, mode="constant", anti_aliasing=False
        ).astype(np.uint8)

        label_ids = np.unique(gt)[1:]
        if len(label_ids) == 0:
            return self.__getitem__((index + 1) % len(self))
        else:
            random_label = np.array([np.random.choice(label_ids)])

            # gt2D = np.uint8(
            #     gt == random.choice(label_ids.tolist())
            # )  # only one label, (1024, 1024)
            #
        try:
            gt2D = np.uint8(gt == random_label)  # only one label, (1024,1024)
        except:
            gt2D = np.uint8(gt == np.max(gt))
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
            "image": torch.tensor(img_1024).float(),
            "gt2D": torch.tensor(gt2D[None, :, :]).long(),
            "bboxes": torch.tensor(bboxes).float(),
            "image_name": img_name
        }