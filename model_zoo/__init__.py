import os
# import clip
# from PIL import Image
# from torchvision import transforms
# from .constants import CACHE_DIR

import csv
import os
from PIL import Image
import torch
# from clip import load
import clip
import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import csv
import sys
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from llm2vec import LLM2Vec
from peft import PeftModel
import torch
from collections import OrderedDict
import torch.nn as nn
import sys
import json
from pathlib import Path

# import braceexpand
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
from torch.utils.data import default_collate
class LLM2CLIP(nn.Module):
    def __init__(self, visual_model=None, text_model=None, adaptor_ckpt=None, is_hf=False):
        super(LLM2CLIP, self).__init__()
        self.visual_model = visual_model
        self.text_model = text_model
        
        self.is_hf = is_hf
        
        # text_embedding_dim = 4096
        # expansion_factor = 2
        # adaptor_num_layers = 4
        # proj_bias = True
        # output_dim = 1280
        # self.text_adaptor = nn.Sequential(
        #     *[LinearBlock(text_embedding_dim, expansion_factor) for _ in range(adaptor_num_layers)],
        #     nn.LayerNorm(text_embedding_dim),
        #     nn.Linear(text_embedding_dim, output_dim, bias=proj_bias),
        # )
        # ckpt = torch.load(adaptor_ckpt)
        # self.text_adaptor.load_state_dict(ckpt)
    
    def encode_image(self, input_pixels):
        if self.is_hf:
           x = self.visual_model.get_image_features(input_pixels)
        else:
            x = self.visual_model.encode_image(input_pixels)
        return x
    
    def encode_text(self, sentence, device='cuda'):
        x = self.text_model.encode(sentence, batch_size=32 ,convert_to_tensor=True).to(device)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        x = self.visual_model.text(x)
        return x
rei_path = '/home/aiscuser/waq/instructCLIP/fusemix/EVA-CLIP/rei'
sys.path.append(rei_path)

class Llama_FeatureExtractor(nn.Module):
    def __init__(self, extra_path=None,):
        super().__init__()
        self.l2v = LLM2Vec.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct",
            "/blob/waq/llm2vec/output/supervised/Meta-Llama-3.2-1B-Instruct_scratch_cc4_5m/E5_train_m-Llama-3.2-1B-Instruct_p-eos_token_b-512_l-512_bidirectional-True_e-1_s-42_w-300_lr-0.0002_lora_r-16/checkpoint-8554",
            merge_peft = True,
            # "/blob/waq/llm2vec/output/mntp/Meta-Llama-3.2-1B-Instruct",
            # extra_model_name_or_path=[
            #     "/blob/waq/llm2vec/output/mntp-supervised/Meta-Llama-3.2-1B-Instruct/E5_train_m-Llama-3.2-1B-Instruct_p-mean_b-512_l-512_bidirectional-True_e-3_s-42_w-300_lr-0.0002_lora_r-16/checkpoint-1000",
            #     "/blob/waq/llm2vec/output/mntp-supervised/Meta-Llama-3.2-1B-Instruct_cc3m/E5_train_m-Llama-3.2-1B-Instruct_p-mean_b-512_l-512_bidirectional-True_e-1_s-42_w-300_lr-0.0002_lora_r-16/checkpoint-8554"
            # ],
            attn_implementation = "flash_attention_2",
            torch_dtype=torch.bfloat16
        )
    def encode(self, *args, **kwargs):
        with torch.cuda.amp.autocast():
            reps = self.l2v.encode(*args, **kwargs)
        reps_norm = torch.nn.functional.normalize(reps, p=2, dim=1)
        return reps_norm
def get_model(model_name, device, root_dir):
    """
    Helper function that returns a model and a potential image preprocessing function.
    """
    if "llm2clip" in model_name.lower():
        from .llm2clip_models import LLM2CLIPWrapper
        text = Llama_FeatureExtractor()
        from eva_clip import create_model_and_transforms, create_model_from_pretrained, trace_model, get_tokenizer
        evamodel, preprocess_train, preprocess_val = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True)

        pretrained = '/home/aiscuser/waq/instructCLIP/new_ckpt_336_1b.pt'
        checkpoint = torch.load(pretrained)
        # checkpoint = checkpoint['module']
        evamodel.load_state_dict(checkpoint, strict=True)
        evamodel = evamodel.to('cuda').eval()
        visual_model = evamodel

        model = LLM2CLIP(visual_model, text)
        model = LLM2CLIPWrapper(model, device)
        image_preprocess = preprocess_val
        return model, image_preprocess
    if "openai-clip" in model_name:
        from .clip_models import CLIPWrapper
        variant = model_name.split(":")[1]
        model, image_preprocess = clip.load(variant, device=device, download_root=root_dir)
        model = model.eval()
        clip_model = CLIPWrapper(model, device) 
        return clip_model, image_preprocess

    elif model_name == "blip-flickr-base":
        from .blip_models import BLIPModelWrapper
        blip_model = BLIPModelWrapper(root_dir=root_dir, device=device, variant="blip-flickr-base")
        image_preprocess = transforms.Compose([
                        transforms.Resize((384, 384),interpolation=transforms.functional.InterpolationMode.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])  
        return blip_model, image_preprocess
    
    elif model_name == "blip-coco-base":
        from .blip_models import BLIPModelWrapper
        blip_model = BLIPModelWrapper(root_dir=root_dir, device=device, variant="blip-coco-base")
        image_preprocess = transforms.Compose([
                        transforms.Resize((384, 384),interpolation=transforms.functional.InterpolationMode.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])  
        return blip_model, image_preprocess
    
    elif model_name == "xvlm-flickr":
        from .xvlm_models import XVLMWrapper
        xvlm_model = XVLMWrapper(root_dir=root_dir, device=device, variant="xvlm-flickr")
        image_preprocess = transforms.Compose([
                            transforms.Resize((384, 384), interpolation=Image.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])
        return xvlm_model, image_preprocess
    
    elif model_name == "xvlm-coco":
        from .xvlm_models import XVLMWrapper
        xvlm_model = XVLMWrapper(root_dir=root_dir, device=device, variant="xvlm-coco")
        image_preprocess = transforms.Compose([
                            transforms.Resize((384, 384), interpolation=Image.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])
        return xvlm_model, image_preprocess
    
    elif model_name == "flava":
        from .flava import FlavaWrapper
        flava_model = FlavaWrapper(root_dir=root_dir, device=device)
        image_preprocess = None
        return flava_model, image_preprocess

    elif model_name == "NegCLIP":
        import open_clip
        from .clip_models import CLIPWrapper
        
        path = os.path.join(root_dir, "negclip.pth")
        if not os.path.exists(path):
            print("Downloading the NegCLIP model...")
            import gdown
            gdown.download(id="1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ", output=path, quiet=False)
        model, _, image_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=path, device=device)
        model = model.eval()
        clip_model = CLIPWrapper(model, device) 
        return clip_model, image_preprocess

    elif model_name == "coca":
        import open_clip
        from .clip_models import CLIPWrapper
        model, _, image_preprocess = open_clip.create_model_and_transforms(model_name="coca_ViT-B-32", pretrained="laion2B-s13B-b90k", device=device)
        model = model.eval()
        clip_model = CLIPWrapper(model, device) 
        return clip_model, image_preprocess
    
        
    elif "laion-clip" in model_name:
        import open_clip
        from .clip_models import CLIPWrapper
        variant = model_name.split(":")[1]
        model, _, image_preprocess = open_clip.create_model_and_transforms(model_name=variant, pretrained="laion2b_s34b_b79k", device=device)
        model = model.eval()
        clip_model = CLIPWrapper(model, device) 
        return clip_model, image_preprocess
    
        
    else:
        raise ValueError(f"Unknown model {model_name}")
