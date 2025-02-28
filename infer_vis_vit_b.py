import copy
import sys

sys.path.append("..")
from glob import glob

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT
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
from segment_anything import sam_model_registry, SamPredictor

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
    default="/data1/hp/promt_sam/once_weight/medsam_vit_b.pth",
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
    '-quan_checkpoint_path',
    type=str,
    default="/data/hp/prompt_medsam/comq_new/quan_vit_b_ckpt/medsam_tinyvit_2bit.pth",
    help='path to the checkpoint of MedSAM-Lite',
)
parser.add_argument(
    '-png_save_dir',
    type=str,
    default='./overlay/CT/vit_b/comq/2bit',
    help='directory to save the overlay image'
)
parser.add_argument(
    '-pred_save_dir',
    type=str,
    default='./pred_save/CT/vit_b/comq/2bit',
    help='directory to save the prediction',
)
parser.add_argument(
    '--overwrite',
    default=True,
    help='whether to overwrite the existing prediction'
)


args = parser.parse_args()

print("======> Set Parameters for Inference")
# dataset_name = args.dataset
data_root = args.data_root
medsam_lite_checkpoint_path = args.medsam_lite_checkpoint_path

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





# %%


medsam_model = sam_model_registry['vit_b'](checkpoint=medsam_lite_checkpoint_path).to(device)
medsam_model.eval()
sam_predictor = SamPredictor(medsam_model)

quan_medsam_model = copy.deepcopy(medsam_model)
quan_checkpoint = torch.load(args.quan_checkpoint_path, map_location='cpu')
quan_medsam_model.image_encoder.load_state_dict(quan_checkpoint)
quan_medsam_model.to(device).eval()
quant_sam_predictor = SamPredictor(quan_medsam_model)


print("======> Start Inference")


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


        for i in range(img_3D.shape[0]):
            img_2d = img_3D[i, :, :]  # (H, W)
            H, W = img_2d.shape[:2]
            img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)  # (H, W, 3)

            sam_predictor.set_image(img_3c)

            img_3c_quant = copy.deepcopy(img_3c)
            quant_sam_predictor.set_image(img_3c_quant)


            gt2D = gt_3D[i, :, :]  # (H, W)
            label_ids = np.unique(gt2D)[1:]
            # print(label_ids)
            for label_id in label_ids:
                gt_label = (gt2D == label_id).astype(np.uint8)
                bbox = get_bbox(gt_label, bbox_shift)
                sam_masks, _, _ = sam_predictor.predict(box=bbox, multimask_output=False)
                seg_3D[i, np.squeeze(sam_masks > 0, axis=0)] = label_id  # 使用 organ_id 保存分割结果

                quant_masks, _, _ = quant_sam_predictor.predict(box=bbox, multimask_output=False)
                quan_sam_3D[i, np.squeeze(quant_masks > 0, axis=0)] = label_id


        np.savez_compressed(
            join(pred_save_dir, task_folder, npz_name),
            sam_seg=seg_3D, quan_sam=quan_sam_3D,gts=gt_3D, spacing=spacing
        )

        # visualize image, mask and bounding box
        if save_overlay:
            idx = int(seg_3D.shape[0] / 2)  # middle slices vis
            fig, ax = plt.subplots(1, 4, figsize=(24, 6))
            ax[0].imshow(img_3D[idx], cmap='gray')
            ax[1].imshow(img_3D[idx], cmap='gray')
            ax[2].imshow(img_3D[idx], cmap='gray')
            ax[3].imshow(img_3D[idx], cmap='gray')
            # ax[4].imshow(img_3D[idx], cmap='gray')
            # ax[5].imshow(img_3D[idx], cmap='gray')

            ax[0].set_title("Image")
            ax[1].set_title("Ground Truth")
            ax[2].set_title("MedSAM")
            ax[3].set_title("MedSAM-Quant")
            # ax[4].set_title(f"MedSAM-Quant_Tunning_2bit")
            # ax[5].set_title("Ours")

            ax[0].axis('off')
            ax[1].axis('off')
            ax[2].axis('off')
            ax[3].axis('off')
            # ax[4].axis('off')
            # ax[5].axis('off')
            for label_id in np.unique(seg_3D[idx])[1:]:
                organ_id = label_map.get(label_id, label_id)
                color = get_label_color(organ_id)

                gt2D = (gt_3D[idx] == label_id).astype(np.uint8)
                box = get_bbox(gt2D)

                show_box(box, ax[0], edgecolor=color)

                show_mask((gt_3D[idx] == label_id).astype(np.uint8), ax[1], mask_color=color)
                show_box(box, ax[1], edgecolor=color)

                show_mask((seg_3D[idx] == label_id).astype(np.uint8), ax[2], mask_color=color)
                show_box(box, ax[2], edgecolor=color)

                show_mask((quan_sam_3D[idx] == label_id).astype(np.uint8), ax[3], mask_color=color)
                show_box(box, ax[3], edgecolor=color)
                #
                # show_mask((quan_tuning_sam_3D[idx] == label_id).astype(np.uint8), ax[4], mask_color=color)
                # show_box(box_viz, ax[4], edgecolor=color)
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
