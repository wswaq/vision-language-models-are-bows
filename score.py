import sys
sys.path.append("/home/aiscuser/waq/instructCLIP/vision-language-models-are-bows")
import pandas as pd

from torch.utils.data import DataLoader
from model_zoo import get_model
from dataset_zoo import VG_Relation, VG_Attribution

root_dir="~/.cache" 
import numpy as np

import os
# os.environ["LLM2VEC_VERSION"] = "3.1_latent_mixd15m"
# model, preprocess = get_model(model_name="llm2clip:ViT-L/14@336px", device="cuda", root_dir=root_dir,pretrained='/blob/hwq/data/tune_logs/T_vitEVA02-CLIP-L-14_32x8*16_lr1e-5_Rd30m_3.1_latent_mixd15m_eval_4ep-2025_01_08-20/checkpoints/epoch_4/mp_rank_00_model_states.pt')
from llm2vec import LLM2Vec
import torch
base_model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
peft_path = "/blob/waq/llm2vec/output/supervised/MetaLlama3.1_scratch_unieos_mix_datacomp15m_e5_im_latent/E5_train_m-Llama-3.1-8B-Instruct_p-latent_b-2048_l-512_bidirectional-False_e-1_s-42_w-300_lr-0.00025_lora_r-16/checkpoint-8092"
# l2v = LLM2Vec.from_pretrained(
# #  "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
# base_model_name_or_path=base_model_name_or_path,
# peft_model_name_or_path=peft_path,
#     # peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
#     device_map="cuda" if torch.cuda.is_available() else "cpu",
#     torch_dtype=torch.bfloat16,
# )
from transformers import AutoModel
l2v = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True,torch_dtype=torch.float16).to("cuda")
class Model_wrapper:
    def __init__(self,model):
        self.model = model
        self.device = "cuda"
    def encode_text(self,text):
        return self.model.encode(text)
# model = Model_wrapper(l2v)
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModel
model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to('cuda')
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

# model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
preprocess = lambda x: processor(images=x, return_tensors="pt", padding=True)
# preprocess = None
# Get the VG-R dataset
vgr_dataset = VG_Relation(image_preprocess=preprocess, download=True, root_dir=root_dir)
vgr_loader = DataLoader(vgr_dataset, batch_size=64, shuffle=False, num_workers=16)

scores = []
from tqdm import tqdm
tqdm_loader = tqdm(vgr_loader)
tqdm_loader.set_description("Computing retrieval scores")
import torch
all_caption_options = []
with torch.no_grad(),torch.cuda.amp.autocast():
    _i=0
    for batch in tqdm_loader:
        image_options = []
        # image_options = torch.stack(batch["image_options"]).to(self.device)
        # origin_shape = image_options.shape

        # # 编码图像
        # image_embeddings = self.model.encode_image(image_options.view(-1, *image_options.shape[2:])).cpu().numpy()  # (B * K) x D
        # image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)  # (B * K) x D
        # image_embeddings = image_embeddings.reshape(origin_shape[0], origin_shape[1], -1)  # B x K x D
        # image_options = image_embeddings
        # for i_option in batch["image_options"]:
        #     image_embeddings = model.model.encode_image(i_option.to('cuda:0')).cpu().numpy() # B x D
        #     image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True) # B x D
        #     image_options.append(np.expand_dims(image_embeddings, axis=1))
        
        caption_options = []
        # texts = []
        # for c_option in batch["caption_options"]:
        #     texts.extend(c_option)
        # caption_embeddings = self.model.encode_text(texts).cpu().numpy() # B x D
        # caption_embeddings = caption_embeddings / np.linalg.norm(caption_embeddings, axis=1, keepdims=True)
        # caption_embeddings = caption_embeddings.reshape(len(batch["caption_options"]), -1, caption_embeddings.shape[-1])
        # caption_options = caption_embeddings
        __i=0 
        for c_option in batch["caption_options"]:

            # import clip
            # caption_tokenized = torch.cat([clip.tokenize(c) for c in c_option])
            texts = c_option
            # print(texts)
            if isinstance(texts, tuple):
                texts = list(texts)
            # texts =caption_tokenized.to('cuda:0')
            try:
                caption_embeddings = model.encode_text(texts).cpu().numpy() # B x D
            except:
                texts = processor(text=texts, return_tensors="pt", padding=True).to('cuda')['input_ids']
                # print(texts)
                caption_embeddings = model.get_text_features(texts).cpu().numpy()
            caption_embeddings = caption_embeddings / np.linalg.norm(caption_embeddings, axis=1, keepdims=True) # B x D
            caption_options.append(np.expand_dims(caption_embeddings, axis=1))
        # image_options = np.concatenate(image_options, axis=1) # B x K x D
        caption_options = np.concatenate(caption_options, axis=1) # B x L x D
        all_caption_options.append(caption_options)
        break
        # print(image_options.shape,caption_options.shape)
        # batch_scores = np.einsum("nkd,nld->nkl", image_options, caption_options) # B x K x L
        # scores.append(batch_scores)
        _i+=1
        # if _i>2:
        #     break
    # print("scores",scores[0].shape)
    # all_scores = np.concatenate(scores, axis=0) # N x K x L
    all_caption_options = np.concatenate(all_caption_options, axis=0) # N x L x D
print("all_caption_options",all_caption_options.shape)
#cal cos between all_caption_options[:,0,:] and all_caption_options[:,1,:]
cos = np.einsum("nd,nd->n", all_caption_options[:,0,:],all_caption_options[:,1,:])
#avg 
print("cos",cos.mean())

 