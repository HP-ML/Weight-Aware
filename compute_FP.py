import torch
import torch.nn as nn
import torch.optim as optim
from fvcore.nn import FlopCountAnalysis
import torch.nn.functional as F
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT
import copy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_tinyvit():
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
    # %%
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

    class MedSAM_Lite(nn.Module):
        def __init__(
                self,
                image_encoder,
                mask_decoder,
                prompt_encoder
        ):
            super().__init__()
            self.image_encoder = image_encoder
            self.mask_decoder = mask_decoder
            self.prompt_encoder = prompt_encoder

        def forward(self, image, box_np):
            image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
            # do not compute gradients for prompt encoder
            with torch.no_grad():
                box_torch = torch.as_tensor(box_np, dtype=torch.float32, device=image.device)
                if len(box_torch.shape) == 2:
                    box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_np,
                masks=None,
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embedding,  # (B, 256, 64, 64)
                image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,
            )  # (B, 1, 256, 256)

            return low_res_masks

        @torch.no_grad()
        def postprocess_masks(self, masks, new_size, original_size):
            """
            Do cropping and resizing

            Parameters
            ----------
            masks : torch.Tensor
                masks predicted by the model
            new_size : tuple
                the shape of the image after resizing to the longest side of 256
            original_size : tuple
                the original shape of the image

            Returns
            -------
            torch.Tensor
                the upsampled mask to the original size
            """
            # Crop
            masks = masks[..., :new_size[0], :new_size[1]]
            # Resize
            masks = F.interpolate(
                masks,
                size=(original_size[0], original_size[1]),
                mode="bilinear",
                align_corners=False,
            )

            return masks

    medsam_lite_model = MedSAM_Lite(
        image_encoder=medsam_lite_image_encoder,
        mask_decoder=medsam_lite_mask_decoder,
        prompt_encoder=medsam_lite_prompt_encoder
    )
    medsam_lite_checkpoint = torch.load('/data1/hp/promt_sam/work_dir/LiteMedSAM/lite_medsam.pth', map_location='cpu')
    medsam_lite_model.load_state_dict(medsam_lite_checkpoint)
    medsam_lite_model = medsam_lite_model
    image_encoder = copy.deepcopy(medsam_lite_model.image_encoder)
    del medsam_lite_model
    return image_encoder
# 初始化模型
model = get_tinyvit()
model.to(device)
# 假设输入的图像大小为 (batch_size, 3, 32, 32)
input_tensor = torch.randn(1, 3, 256, 256)  # batch_size=1，3通道，32x32图像
input_tensor = input_tensor.to(device)
# 计算FLOPs
flops = FlopCountAnalysis(model, input_tensor)
print(f"模型的FLOPs: {flops.total()}")

# 计算显存使用情况
# 打印模型参数的显存占用
total_params = sum(p.numel() for p in model.parameters())
param_memory = total_params * 4 / 1024 / 1024  # 每个参数占用4字节 (32位浮动精度)，转换为MB
print(f"模型参数的显存占用: {param_memory:.2f} MB")

# 计算中间激活的显存占用
# 假设输入尺寸为 (1, 3, 32, 32)，此时卷积后尺寸会变成 (1, 32, 8, 8)，fc层前的展平尺寸为 32 * 8 * 8
activation_memory = 1 * 32 * 8 * 8 * 4 / 1024 / 1024  # 计算中间激活占用（以MB为单位）
print(f"中间激活的显存占用: {activation_memory:.2f} MB")

# 打印显存使用情况（PyTorch显存）
import torch.cuda
if torch.cuda.is_available():
    print(f"当前CUDA显存分配: {torch.cuda.memory_allocated()} 字节")
    print(f"当前CUDA显存保留: {torch.cuda.memory_reserved()} 字节")
else:
    print("CUDA不可用")


