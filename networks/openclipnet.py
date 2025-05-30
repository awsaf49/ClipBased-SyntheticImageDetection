'''                                        
Copyright 2024 Image Processing Research Group of University Federico
II of Naples ('GRIP-UNINA'). All rights reserved.
                        
Licensed under the Apache License, Version 2.0 (the "License");       
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at                    
                                           
    http://www.apache.org/licenses/LICENSE-2.0
                                                      
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,    
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                         
See the License for the specific language governing permissions and
limitations under the License.
'''

import os
# Set Hugging Face cache directory - change this path as needed
os.environ['HF_HOME'] = './hf_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = './hf_cache'

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import timm
from .resnet_mod import ChannelLinear

dict_pretrain = {
    # CLIP models
    'clipL14openai'     : ('ViT-L-14', 'openai'),
    'clipL14laion400m'  : ('ViT-L-14', 'laion400m_e32'),
    'clipL14laion2B'    : ('ViT-L-14', 'laion2b_s32b_b82k'),
    'clipL14datacomp'   : ('ViT-L-14', 'laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K', 'open_clip_pytorch_model.bin'),
    'clipL14commonpool' : ('ViT-L-14', "laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K", 'open_clip_pytorch_model.bin'),
    'clipaL14datacomp'  : ('ViT-L-14-CLIPA', 'datacomp1b'),
    'cocaL14laion2B'    : ('coca_ViT-L-14', 'laion2b_s13b_b90k'),
    'clipg14laion2B'    : ('ViT-g-14', 'laion2b_s34b_b88k'),
    'eva2L14merged2b'   : ('EVA02-L-14', 'merged2b_s4b_b131k'),
    'clipB16laion2B'    : ('ViT-B-16', 'laion2b_s34b_b88k'),
    'siglip2-l6-256'    : ('ViT-L-16-SigLIP2-256', 'webli'),
    'siglip-l6-256'    : ('ViT-L-16-SigLIP-256', 'webli'),
    
    # ImageNet-trained ViT models from timm
    'vitl16_in21k'      : ('vit_large_patch16_224.orig_in21k', None),  # ImageNet-21k pretrained
    'vitl16_in21k_ft_in1k' : ('vit_large_patch16_224.augreg_in21k_ft_in1k', None),  # IN-21k pretrained + IN-1k finetuned
    'vitl16_mae'        : ('vit_large_patch16_224.mae', None),  # MAE pretrained
    'vitb16_in21k'      : ('vit_base_patch16_224.orig_in21k', None),  # ImageNet-21k pretrained
    'vitb16_in21k_ft_in1k' : ('vit_base_patch16_224.orig_in21k_ft_in1k', None),  # IN-21k pretrained + IN-1k finetuned
    'vitb16_dino'       : ('vit_base_patch16_224.dino', None),  # DINO self-supervised
    'vitb16_mae'        : ('vit_base_patch16_224.mae', None),  # MAE pretrained
    'vitb16_clip'       : ('vit_base_patch16_clip_224.openai', None),  # OpenAI CLIP pretrained
    'vitb16_clip_ft_in1k' : ('vit_base_patch16_clip_224.openai_ft_in1k', None),  # CLIP pretrained + IN-1k finetuned
}

class OpenClipLinear(nn.Module):
    def __init__(self, num_classes=1, pretrain='clipL14commonpool', normalize=True, next_to_last=False):
        super(OpenClipLinear, self).__init__()
        
        model_config = dict_pretrain[pretrain]
        model_name = model_config[0]
        
        # Check if this is a timm model (ImageNet pretrained) or CLIP model
        if model_name.startswith('vit_'):
            # Initialize timm ViT model
            backbone = timm.create_model(model_name, pretrained=True, num_classes=0)  # num_classes=0 removes classifier
            self.is_clip_model = False
        else:
            # Initialize CLIP model
            if len(model_config) == 2:
                backbone = open_clip.create_model(model_config[0], pretrained=model_config[1])
            else:
                from huggingface_hub import hf_hub_download
                backbone = open_clip.create_model(model_config[0], pretrained=hf_hub_download(*model_config[1:]))
            self.is_clip_model = True
        
        # Handle different model types to get feature dimension
        if self.is_clip_model:
            if hasattr(backbone.visual, 'output_dim'):
                # Standard OpenCLIP model
                if next_to_last:
                    self.num_features = backbone.visual.proj.shape[0]
                    backbone.visual.proj = None
                else:
                    self.num_features = backbone.visual.output_dim
            elif hasattr(backbone.visual, 'trunk') and hasattr(backbone.visual.trunk, 'embed_dim'):
                # TimmModel - use trunk embed_dim
                self.num_features = backbone.visual.trunk.embed_dim
                if next_to_last and hasattr(backbone.visual, 'proj'):
                    backbone.visual.proj = None
            elif hasattr(backbone, 'embed_dim'):
                # Other models with embed_dim
                self.num_features = backbone.embed_dim
                if next_to_last and hasattr(backbone.visual, 'proj'):
                    backbone.visual.proj = None
            else:
                # Fallback - try to infer from forward pass
                try:
                    with torch.no_grad():
                        dummy_input = torch.randn(1, 3, 224, 224)
                        features = backbone.encode_image(dummy_input)
                        self.num_features = features.shape[-1]
                except Exception as e:
                    raise ValueError(f"Cannot determine feature dimensions for model type: {type(backbone)}. Error: {e}")
        else:
            # For timm models, feature dimension is in num_features
            self.num_features = backbone.num_features
        
        self.bb = [backbone, ]
        self.normalize = normalize
        
        self.fc = ChannelLinear(self.num_features, num_classes)
        torch.nn.init.normal_(self.fc.weight.data, 0.0, 0.02)

    def to(self, *args, **kwargs):
        self.bb[0].to(*args, **kwargs)
        super(OpenClipLinear, self).to(*args, **kwargs)
        return self

    def forward_features(self, x):
        with torch.no_grad():
            self.bb[0].eval()
            if self.is_clip_model:
                features = self.bb[0].encode_image(x, normalize=self.normalize)
            else:
                features = self.bb[0](x)  # timm models directly output features
                if self.normalize:
                    features = F.normalize(features, p=2, dim=-1)
        return features

    def forward_head(self, x):
        return self.fc(x)

    def forward(self, x):
        return self.forward_head(self.forward_features(x))
