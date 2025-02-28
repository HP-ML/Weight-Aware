import copy
import sys

sys.path.append("..")
from glob import glob

# from auto_promt_model.TokenToToken import DiffusionModel
# from compare_model.Prototype_Prompt_Encoder import Prototype_Prompt_Encoder, Learnable_Prototypes
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT
# from compare_model.model_forward import model_forward_function, postprocess_masks
import argparse
import numpy as np
import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torch.multiprocessing as mp
from multiprocessing import Manager
from os import makedirs
from tqdm import tqdm
from os import listdir, makedirs
from os.path import join, exists, isfile, isdir, basename
# /data1/hp/medsamdata/CT/split_data/AbdomenCT1K/valid
# /data1/hp/medsamdata/MR/split_data/BraTS_FLAIR/valid
print("======> Process Arguments")
parser = argparse.ArgumentParser()
parser.add_argument(
    '-data_root',
    type=str,
    default='/data1/hp/medsamdata/CT/split_data/AbdomenCT1K/valid',
    help='root directory of the data',
)

parser.add_argument(
    '-medsam_lite_checkpoint_path',
    type=str,
    default="../work_dir/LiteMedSAM/lite_medsam.pth",
    help='path to the checkpoint of MedSAM-Lite',
)
# /data/hp/prompt_medsam/workdir/CT_AbdomenCT1K_Prototype_best.pth
# MR_BraTS_Prototype_best.pth
# XRay_Chest_Prototype_best.pth
# Endoscopy_Kvasir_Prototype_best.pth
# parser.add_argument(
#     '-CT_Prototype_checkpoint_path',
#     type=str,
#     default="/data/hp/prompt_medsam/workdir/CT_AbdomenCT1K_Prototype_best.pth",
#     help='path to the checkpoint of MedSAM-Lite',
# )
# /data1/hp/promt_sam/once_weight
# work_dir/LiteMedSAM/lite_medsam.pth
parser.add_argument(
    '-quan_checkpoint_path',
    type=str,
    default="/data/hp/prompt_medsam/comq_new/quan_ckpt/medsam_tinyvit_4bit.pth",
    help='path to the checkpoint of MedSAM-Lite',
)
parser.add_argument(
    '-quan_tunning_checkpoint_path',
    type=str,
    default="/data/hp/prompt_medsam/comq_new/quan_ckpt/batchtunning/medsam_tinyvit_4bit.pth",
    help='path to the checkpoint of MedSAM-Lite',
)
parser.add_argument(
    '-device',
    type=str,
    default="cuda:0",
    help='device to run the inference',
)
parser.add_argument(
    '-num_workers',
    type=int,
    default=4,
    help='number of workers for inference with multiprocessing',
)
parser.add_argument(
    '--save_overlay',
    default=True,
    help='whether to save the overlay image'
)
parser.add_argument(
    '-png_save_dir',
    type=str,
    default='./overlay/CT/4bit',
    help='directory to save the overlay image'
)
parser.add_argument(
    '-pred_save_dir',
    type=str,
    default='./pred_save/CT/4bit',
    help='directory to save the prediction',
)
parser.add_argument(
    '--overwrite',
    default=True,
    help='whether to overwrite the existing prediction'
)
# parser.add_argument(
#     '-medsam_lite_fine_tune_checkpoint_path',
#     type=str,
#     default="/data/hp/prompt_medsam/workdir/medsam_fine_tune_best.pth",
#     help='path to the checkpoint of MedSAM-Lite',
# )
# /data1/hp/promt_sam/workdir/CT_AbdomenCT1K_best.pth
# MR_BraTS_best.pth
# XRay_chest_best.pth
# Endoscopy_Kvasir_best.pth
# parser.add_argument(
#     '-auto_medsam_checkpoint_path',
#     type=str,
#     default="/data1/hp/promt_sam/workdir/CT_AbdomenCT1K_best.pth",
#     help='path to the checkpoint of MedSAM-Lite',
# )

args = parser.parse_args()

print("======> Set Parameters for Inference")
# dataset_name = args.dataset
data_root = args.data_root
medsam_lite_checkpoint_path = args.medsam_lite_checkpoint_path
# medsam_lite_fine_tune_checkpoint_path = args.medsam_lite_fine_tune_checkpoint_path
# CT_Prototype_checkpoint_path = args.CT_Prototype_checkpoint_path
# auto_medsam_checkpoint_path = args.auto_medsam_checkpoint_path

pred_save_dir = args.pred_save_dir
save_overlay = args.save_overlay
num_workers = args.num_workers
overwrite = args.overwrite

manager = Manager()  # MODIFIED: Add manager for multiprocessing shared dict
label_colors = manager.dict()
makedirs(pred_save_dir, exist_ok=True)
bbox_shift = 5
device = torch.device(args.device)
gt_path_files = sorted(glob(join(data_root, '*.npz'), recursive=True))
image_size = 256

# CT_Abdomen 15
label_map = {
    1: 1, 3: 2, 4: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9,
    15: 10, 16: 11, 17: 12, 19: 13, 20: 14, 22: 15
}
# # MR_BraTs 1
# # self.label_map = {
# #     1: 1
# # }
# # XRay_Chest 10
# self.label_map = {
#     2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 12: 8, 13: 9, 255: 10
# }
# # Endoscopy_Kvasir 10
# # self.label_map = {
# #     2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 9: 7, 10: 8, 15: 9, 16: 10
# #
# label_map = {
#             1: 1, 3: 2, 4: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9,
#             15: 10, 16: 11, 17: 12, 19: 13, 20: 14, 22: 15
#         }
# label_map = {
#     1: 1
# }


if save_overlay:
    assert args.png_save_dir is not None, "Please specify the directory to save the overlay image"
    png_save_dir = args.png_save_dir
    makedirs(png_save_dir, exist_ok=True)

print("======> Load SAM")


class MedSAM_Lite(nn.Module):
    def __init__(self,
                 image_encoder,
                 mask_decoder,
                 prompt_encoder
                 ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image, boxes):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )  # (B, 1, 256, 256)

        return low_res_masks, iou_predictions

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Do cropping and resizing
        """
        # Crop
        masks = masks[:, :, :new_size[0], :new_size[1]]
        # Resize
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        return masks


# %%
medsam_lite_image_encoder = TinyViT(
    img_size=256,
    in_chans=3,
    embed_dims=[
        64,  ## (64, 256, 256)
        128,  ## (128, 128, 128)
        160,  ## (160, 64, 64)
        320  ## (320, 64, 64)
    ],
    depths=[2, 2, 6, 2],
    num_heads=[2, 4, 5, 10],
    window_sizes=[7, 7, 14, 7],
    mlp_ratio=4.,
    drop_rate=0.,
    drop_path_rate=0.0,
    use_checkpoint=False,
    mbconv_expand_ratio=4.0,
    local_conv_size=3,
    layer_lr_decay=0.8
)

medsam_lite_prompt_encoder = PromptEncoder(
    embed_dim=256,
    image_embedding_size=(64, 64),
    input_image_size=(256, 256),
    mask_in_chans=16
)

medsam_lite_mask_decoder = MaskDecoder(
    num_multimask_outputs=3,
    transformer=TwoWayTransformer(
        depth=2,
        embedding_dim=256,
        mlp_dim=2048,
        num_heads=8,
    ),
    transformer_dim=256,
    iou_head_depth=3,
    iou_head_hidden_dim=256,
)

medsam_lite_model = MedSAM_Lite(
    image_encoder=medsam_lite_image_encoder,
    mask_decoder=medsam_lite_mask_decoder,
    prompt_encoder=medsam_lite_prompt_encoder
)

quan_medsam_lite_model = MedSAM_Lite(
    image_encoder=copy.deepcopy(medsam_lite_image_encoder),
    mask_decoder=copy.deepcopy(medsam_lite_mask_decoder),
    prompt_encoder=copy.deepcopy(medsam_lite_prompt_encoder)
)


# class MedSAM_Lite_With_Diffusion(MedSAM_Lite):
#     def __init__(self, image_encoder, mask_decoder, prompt_encoder, diffusion_model):
#         super().__init__(image_encoder, mask_decoder, prompt_encoder)
#         self.diffusion_model = diffusion_model
#
#     def forward(self, image, boxes, label):
#         # Step 1: Encode the image
#         image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
#
#         # Step 2: Use diffusion model to generate sparse embeddings
#         sparse_embeddings = self.diffusion_model(image_embedding, label)  # (B, 2, 256)
#
#         # Step 3: Get dense embeddings from prompt encoder
#         prpmpt_sparse_embeddings, dense_embeddings = self.prompt_encoder(
#             points=None,
#             boxes=boxes,
#             masks=None,
#         )
#         # no_mask_embed = self.prompt_encoder.no_mask_embed
#         # dense_embeddings = no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
#         #     batch_size, -1, 64, 64)
#
#         # Step 4: Decode the mask using the mask decoder
#         low_res_masks, iou_predictions = self.mask_decoder(
#             image_embeddings=image_embedding,  # (B, 256, 64, 64)
#             image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
#             sparse_prompt_embeddings=sparse_embeddings,  # Adjusted shape (B, 2, 256)
#             dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
#             multimask_output=False,
#         )  # (B, 1, 256, 256)
#
#         return low_res_masks, iou_predictions, sparse_embeddings, prpmpt_sparse_embeddings


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_256, new_size, original_size):
    box_torch = torch.as_tensor(box_256[None, None, ...], dtype=torch.float, device=img_embed.device)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False
    )

    low_res_pred = medsam_model.postprocess_masks(low_res_logits, new_size, original_size)
    low_res_pred = torch.sigmoid(low_res_pred)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)

    return medsam_seg


# @torch.no_grad()
# def medsam_auto_inference(auto_medsam_model, img_embed, box_256, new_size, original_size, label):
#     box_torch = torch.as_tensor(box_256[None, None, ...], dtype=torch.float, device=img_embed.device)
#
#     sparse_embeddings = auto_medsam_model.diffusion_model(img_embed, label)
#
#     prpmpt_sparse_embeddings, dense_embeddings = auto_medsam_model.prompt_encoder(
#         points=None,
#         boxes=box_torch,
#         masks=None,
#     )
#     low_res_logits, _ = auto_medsam_model.mask_decoder(
#         image_embeddings=img_embed,  # (B, 256, 64, 64)
#         image_pe=auto_medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
#         sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
#         dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
#         multimask_output=False
#     )
#
#     low_res_pred = auto_medsam_model.postprocess_masks(low_res_logits, new_size, original_size)
#     low_res_pred = torch.sigmoid(low_res_pred)
#     low_res_pred = low_res_pred.squeeze().cpu().numpy()
#     medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
#
#     return medsam_seg


# token_generate_model = DiffusionModel(embedding_channels=256, timesteps=100)
# medsam_with_auto_prompt = MedSAM_Lite_With_Diffusion(image_encoder=copy.deepcopy(medsam_lite_image_encoder),
#                                                      mask_decoder=copy.deepcopy(medsam_lite_mask_decoder),
#                                                      prompt_encoder=copy.deepcopy(medsam_lite_prompt_encoder),
#                                                      diffusion_model=token_generate_model)
# auto_medsam_lite_checkpoint = torch.load(auto_medsam_checkpoint_path, map_location='cpu')
# medsam_with_auto_prompt.load_state_dict(auto_medsam_lite_checkpoint['model'])
# medsam_with_auto_prompt.to(device)
# medsam_with_auto_prompt.eval()

# medsam_lite_fine_tune_checkpoint = torch.load(medsam_lite_fine_tune_checkpoint_path, map_location='cpu')
# medsam_lite_fine_tune_model.load_state_dict(medsam_lite_fine_tune_checkpoint['model'])
# medsam_lite_fine_tune_model.to(device)
# medsam_lite_fine_tune_model.eval()

medsam_lite_checkpoint = torch.load(medsam_lite_checkpoint_path, map_location='cpu')
medsam_lite_model.load_state_dict(medsam_lite_checkpoint)
medsam_lite_model.to(device)
medsam_lite_model.eval()

quan_medsam_lite_model = copy.deepcopy(medsam_lite_model)
quan_tinyvit_checkpoint = torch.load(args.quan_checkpoint_path, map_location='cpu')
# quan_medsam_lite_model.load_state_dict(medsam_lite_checkpoint)
quan_medsam_lite_model.image_encoder.load_state_dict(quan_tinyvit_checkpoint)
quan_medsam_lite_model.to(device)
quan_medsam_lite_model.eval()

quan_medsam_lite_tunning_model  = copy.deepcopy(medsam_lite_model)
quan_tunning_checkpoint = torch.load(args.quan_tunning_checkpoint_path, map_location='cpu')
quan_medsam_lite_tunning_model.image_encoder.load_state_dict(quan_tunning_checkpoint)
quan_medsam_lite_tunning_model.to(device)
quan_medsam_lite_tunning_model.eval()

# sam_prompt_encoder = copy.deepcopy(medsam_lite_model.prompt_encoder)
# sam_decoder = medsam_lite_model.mask_decoder
# sam_decoder = copy.deepcopy(medsam_lite_mask_decoder)

# sam_prompt_encoder, sam_decoder = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam_prompt_encoder.to(device)
# sam_decoder.to(device)

# print("======> Load Prototypes and Prototype-based Prompt Encoder")
# define the models
# learnable_prototypes_model = Learnable_Prototypes(num_classes=15, feat_dim=256).to(device)
# protoype_prompt_encoder = Prototype_Prompt_Encoder(feat_dim=256,
#                                                    hidden_dim_dense=128,
#                                                    hidden_dim_sparse=128,
#                                                    size=64,
#                                                    num_tokens=2).to(device)

# load the weight for prototype-based prompt encoder, mask decoder, and prototypes
# checkpoint = torch.load(CT_Prototype_checkpoint_path)
# protoype_prompt_encoder.load_state_dict(checkpoint['prototype_prompt_encoder_state_dict'])
# sam_decoder.load_state_dict(checkpoint['sam_decoder_state_dict'])
# learnable_prototypes_model.load_state_dict(checkpoint['prototypes_state_dict'])

# set requires_grad to False to the whole model
# for name, param in sam_prompt_encoder.named_parameters():
#     param.requires_grad = False
# for name, param in sam_decoder.named_parameters():
#     param.requires_grad = False
# for name, param in protoype_prompt_encoder.named_parameters():
#     param.requires_grad = False
# for name, param in learnable_prototypes_model.named_parameters():
#     param.requires_grad = False

print("======> Start Inference")
# binary_masks = dict()
# protoype_prompt_encoder.eval()
# sam_decoder.eval()
# learnable_prototypes_model.eval()


def show_mask(mask, ax, mask_color=None, alpha=0.5):
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, edgecolor='blue'):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0, 0, 0, 0), lw=2))


def resize_box(box, new_size, original_size):
    """
    Revert box coordinates from scale at 256 to original scale

    Parameters
    ----------
    box : np.ndarray
        box coordinates at 256 scale
    new_size : tuple
        Image shape with the longest edge resized to 256
    original_size : tuple
        Original image shape

    Returns
    -------
    np.ndarray
        box coordinates at original scale
    """
    new_box = np.zeros_like(box)
    ratio = max(original_size) / max(new_size)
    for i in range(len(box)):
        new_box[i] = int(box[i] * ratio)

    return new_box


def get_bbox(gt2D, bbox_shift=5):
    assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, f'ground truth should be 0, 1, but got {np.unique(gt2D)}'
    y_indices, x_indices = np.where(gt2D > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = gt2D.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H, y_max + bbox_shift)
    bboxes = np.array([x_min, y_min, x_max, y_max])

    return bboxes


# def get_label_color(label_id):
#     """Get or generate a consistent color for each label."""
#     if label_id not in label_colors:
#         # Generate a new random color for the new label
#         label_colors[label_id] = np.random.rand(3)
#     return label_colors[label_id]
color_index = 0
def get_label_color(label_id):
    """Get a consistent color for each label from a predefined list."""
    predefined_colors = [
        [0.85, 0.1, 0.1],    # Red
        [0.1, 0.85, 0.1],    # Green
        [0.1, 0.1, 0.85],    # Blue
        [0.85, 0.85, 0.1],   # Yellow
        [0.85, 0.1, 0.85],   # Magenta
        [0.1, 0.85, 0.85],   # Cyan
        [0.6, 0.3, 0.1],     # Brown
        [0.5, 0.5, 0.1],     # Olive
        [0.3, 0.6, 0.6],     # Teal
        [0.9, 0.4, 0.1],     # Orange
        [0.5, 0.1, 0.7],     # Purple
        [0.2, 0.7, 0.5],     # Sea Green
        [0.9, 0.2, 0.5],     # Rose
        [0.7, 0.7, 0.7],     # Gray
        [0.3, 0.3, 0.9]      # Light Blue
    ]
    global color_index
    # Use a dictionary to store label-to-color mappings
    if label_id not in label_colors:
        if color_index < len(predefined_colors):
            label_colors[label_id] = predefined_colors[color_index]
            color_index += 1
        else:
            label_colors[label_id] = np.random.rand(3)
        # color_idx = len(label_colors) % len(predefined_colors)
        # label_colors[label_id] = predefined_colors[color_idx]
    return label_colors[label_id]



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


def MedSAM_infer_npz(gt_path_file):
    # prototypes = learnable_prototypes_model()
    npz_name = basename(gt_path_file)
    task_folder = gt_path_file.split('/')[-3]
    makedirs(join(pred_save_dir, task_folder), exist_ok=True)
    if (not isfile(join(pred_save_dir, task_folder, npz_name))) or overwrite:
        npz_data = np.load(gt_path_file, 'r', allow_pickle=True)  # (H, W, 3)
        img_3D = npz_data['imgs']  # (Num, H, W)
        gt_3D = npz_data['gts']  # (Num, H, W)
        spacing = npz_data['spacing']
        seg_3D = np.zeros_like(gt_3D, dtype=np.uint8)  # (Num, H, W)
        quan_sam_3D = np.zeros_like(gt_3D, dtype=np.uint8)  # (Num, H, W)
        quan_tuning_sam_3D = np.zeros_like(gt_3D, dtype=np.uint8)  # (Num, H, W)

        # seg_sur_3D = np.zeros_like(gt_3D, dtype=np.uint8)  # (Num, H, W)
        # auto_seg_3D = np.zeros_like(gt_3D, dtype=np.uint8)  # (Num, H, W)

        box_list = [dict() for _ in range(img_3D.shape[0])]

        for i in range(img_3D.shape[0]):
            img_2d = img_3D[i, :, :]  # (H, W)
            H, W = img_2d.shape[:2]
            img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)  # (H, W, 3)

            ## MedSAM Lite preprocessing
            img_256 = resize_longest_side(img_3c, 256)
            newh, neww = img_256.shape[:2]
            img_256 = (img_256 - img_256.min()) / np.clip(
                img_256.max() - img_256.min(), a_min=1e-8, a_max=None
            )
            img_256_padded = pad_image(img_256, 256)
            img_256_tensor = torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                image_embedding = medsam_lite_model.image_encoder(img_256_tensor)
                quan_image_embedding = quan_medsam_lite_model.image_encoder(img_256_tensor)
                quan_tuning_image_embedding = quan_medsam_lite_tunning_model.image_encoder(img_256_tensor)
                # fine_image_embedding = medsam_lite_fine_tune_model.image_encoder(img_256_tensor)

                # auto_image_embedding = medsam_with_auto_prompt.image_encoder(img_256_tensor)

            gt = gt_3D[i, :, :]  # (H, W)
            label_ids = np.unique(gt)[1:]
            # print(label_ids)
            for label_id in label_ids:
                # Assign a consistent color for each label if not already assigned
                # color = get_label_color(label_id)
                gt2D = np.uint8(gt == label_id)  # only one label, (H, W)
                if gt2D.shape != (newh, neww):
                    gt2D_resize = cv2.resize(
                        gt2D.astype(np.uint8), (neww, newh),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(np.uint8)
                else:
                    gt2D_resize = gt2D.astype(np.uint8)
                gt2D_padded = pad_image(gt2D_resize, 256)  ## (256, 256)
                if np.sum(gt2D_padded) > 0:
                    random_label_mapped = label_map.get(label_id, label_id)
                    # cls_id = torch.tensor(np.array([random_label_mapped])).long().to(device)
                    # preds, preds_quality = model_forward_function(protoype_prompt_encoder, sam_prompt_encoder,
                    #                                               sam_decoder,
                    #                                               image_embedding, prototypes, cls_id)
                    # low_res_pred = postprocess_masks(preds, (newh, neww), (H, W))
                    # low_res_pred = torch.sigmoid(low_res_pred)
                    # low_res_pred = low_res_pred.squeeze().cpu().numpy()
                    # sur_mask = (low_res_pred > 0.5).astype(np.uint8)
                    # seg_sur_3D[i, sur_mask > 0] = label_id

                    box = get_bbox(gt2D_padded, bbox_shift)  # (4,)
                    sam_mask = medsam_inference(medsam_lite_model, image_embedding, box, (newh, neww), (H, W))
                    seg_3D[i, sam_mask > 0] = label_id

                    # quan
                    quan_mask = medsam_inference(quan_medsam_lite_model, quan_image_embedding, box, (newh, neww), (H, W))
                    quan_sam_3D[i, quan_mask > 0] = label_id

                    # quan tunning
                    quan_tunning_mask = medsam_inference(quan_medsam_lite_tunning_model, quan_tuning_image_embedding, box, (newh, neww), (H, W))
                    quan_tuning_sam_3D[i, quan_tunning_mask > 0] = label_id

                    # fine_sam_mask = medsam_inference(medsam_lite_fine_tune_model, fine_image_embedding, box,
                    #                                  (newh, neww), (H, W))
                    # sam_fine_3D[i, fine_sam_mask > 0] = label_id

                    # object_label = torch.tensor(np.array([label_id])).to(device)
                    # auto_sam_mask = medsam_auto_inference(medsam_with_auto_prompt, auto_image_embedding, box,
                    #                                       (newh, neww), (H, W), object_label)
                    # auto_seg_3D[i, auto_sam_mask > 0] = label_id

                    box_list[i][label_id] = box

        label_ids = np.unique(gt_3D)[1:]

        np.savez_compressed(
            join(pred_save_dir, task_folder, npz_name),
            sam_seg=seg_3D, quan_sam=quan_sam_3D, quan_tunning_sam = quan_tuning_sam_3D,gts=gt_3D, spacing=spacing
        )

        # visualize image, mask and bounding box
        if save_overlay:
            idx = int(seg_3D.shape[0] / 2)  # middle slices vis
            box_dict = box_list[idx]
            fig, ax = plt.subplots(1, 5, figsize=(30, 6))
            ax[0].imshow(img_3D[idx], cmap='gray')
            ax[1].imshow(img_3D[idx], cmap='gray')
            ax[2].imshow(img_3D[idx], cmap='gray')
            ax[3].imshow(img_3D[idx], cmap='gray')
            ax[4].imshow(img_3D[idx], cmap='gray')
            # ax[5].imshow(img_3D[idx], cmap='gray')

            ax[0].set_title("Image")
            ax[1].set_title("Ground Truth")
            ax[2].set_title("MedSAM")
            ax[3].set_title("MedSAM-Quant_2bit")
            ax[4].set_title(f"MedSAM-Quant_Tunning_2bit")
            # ax[5].set_title("Ours")

            ax[0].axis('off')
            ax[1].axis('off')
            ax[2].axis('off')
            ax[3].axis('off')
            ax[4].axis('off')
            # ax[5].axis('off')
            for label_id, box_256 in box_dict.items():
                organ_id = label_map.get(label_id, label_id)
                color = get_label_color(organ_id)
                # purple_color = [0.9, 0.2, 0.5]
                # color = np.random.rand(3)
                box_viz = resize_box(box_256, (newh, neww), (H, W))
                show_box(box_viz, ax[0], edgecolor=color)

                show_mask((gt_3D[idx] == label_id).astype(np.uint8), ax[1], mask_color=color)
                show_box(box_viz, ax[1], edgecolor=color)

                show_mask((seg_3D[idx] == label_id).astype(np.uint8), ax[2], mask_color=color)
                show_box(box_viz, ax[2], edgecolor=color)

                show_mask((quan_sam_3D[idx] == label_id).astype(np.uint8), ax[3], mask_color=color)
                show_box(box_viz, ax[3], edgecolor=color)
                #
                show_mask((quan_tuning_sam_3D[idx] == label_id).astype(np.uint8), ax[4], mask_color=color)
                show_box(box_viz, ax[4], edgecolor=color)
                #
                # show_mask((auto_seg_3D[idx] == label_id).astype(np.uint8), ax[5], mask_color=color)
                # show_box(box_viz, ax[2], edgecolor=color)

                # show_mask(auto_seg_3D[idx], ax[3], mask_color=color)
            plt.tight_layout()
            plt.savefig(join(png_save_dir, npz_name.split(".")[0] + '.png'), dpi=300)
            plt.close()


if __name__ == '__main__':
    for gt_path in gt_path_files:
        MedSAM_infer_npz(gt_path)
