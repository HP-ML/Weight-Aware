import copy
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import torch.nn as nn

from quant import *

import timm


DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def find_layers(module, layers=[nn.Conv2d, nn.Linear, ActQuantWrapper], name=''):
    if type(module) in layers:
        
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        # print(name1)
        if 'embed' not in name1 or 'head' not in name1:
            res.update(find_layers(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1
            ))
    return res

def freeze_layers_tinyvit(model):
    from tiny_vit_sam import PatchEmbed, MBConv, PatchMerging, ConvLayer, Mlp, Attention, TinyViTBlock, BasicLayer, LayerNorm2d, Conv2d_BN, DropPath
    for layer in model.children():
        if type(layer) in {nn.Sequential, nn.ModuleList, PatchEmbed, MBConv, PatchMerging, ConvLayer, Mlp, Attention, TinyViTBlock, BasicLayer, LayerNorm2d, Conv2d_BN, DropPath}:
            freeze_layers_tinyvit(layer)
        else:
            if type(layer) == nn.Conv2d:
                layer.weight.requires_grad = False
                # layer.bias.requires_grad = False
            elif type(layer) == nn.Linear:
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False

@torch.no_grad()
def test(model, dataloader):
    train = model.training
    model.eval()
    print('Evaluating ...')
    dev = next(iter(model.parameters())).device
    preds = []
    ys = []
    for x, y in dataloader:
        preds.append(torch.argmax(model(x.to(dev)), 1))
        ys.append(y.to(dev))
    acc = torch.mean((torch.cat(preds) == torch.cat(ys)).float()).item()
    acc *= 100
    print('%.2f' % acc)
    if model.training:
        model.train()

def get_test(name):

    return test


def run(model, batch, loss=False, retmoved=False):
    images = batch['image']
    dev = next(iter(model.parameters())).device
    # if retmoved:
    #     return (images.to(dev), gts.to(dev))
    out = model(images.to(dev))
    if loss:
        return nn.functional.cross_entropy(out, batch[1].to(dev)).item() * batch[0].shape[0]
    return out

def get_run(model):

    return run


def get_vit():
    model_name = 'vit_small_patch16_224'
    model = timm.create_model(model_name, pretrained=True)
    model = model.to(DEV)

    return model
def get_tinyvit():
    from tiny_vit_sam import TinyViT
    import torch.nn.functional as F
    from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
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
    return medsam_lite_model, image_encoder

def get_vit_b():
    from segment_anything import sam_model_registry
    medsam_checkpoint = "/data1/hp/promt_sam/once_weight/medsam_vit_b.pth"
    sam_model = sam_model_registry["vit_b"](checkpoint=medsam_checkpoint)
    image_encoder = copy.deepcopy(sam_model.image_encoder)
    return sam_model, image_encoder


from torchvision.models import resnet18, resnet34, resnet50, resnet101 

get_models = {
    'rn18': lambda: resnet18(pretrained=True),
    'rn34': lambda: resnet34(pretrained=True),
    'rn50': lambda: resnet50(pretrained=True),
    'vit_small': lambda: get_vit(),
    'tiny_vit': lambda: get_tinyvit(),
    'vit_b': lambda: get_vit_b(),
}

def get_model(model):
    medsam, model = get_models[model]()
    model = model.to(DEV)
    model.eval()
    return medsam, model


def get_functions(model):
    return lambda: get_model(model), get_test(model), get_run(model)


def firstlast_names(model):
    if 'rn' in model:
        return ['conv1', 'fc']
    if 'bertsquad' in model:
        return [
            'bert.embeddings.word_embeddings',
            'bert.embeddings.token_type_embeddings',
            'qa_outputs'
        ]
    if 'yolo' in model:
        lastidx = {'n': 24}[model[6]]
        return ['model.0.conv'] + ['model.%d.m.%d' % (lastidx, i) for i in range(3)]
