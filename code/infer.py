# The code is adopted frin Zirui's code (https://raw.githubusercontent.com/LostPetInitiative/study_spring_2022/main/zhirui/infer.py)

import os.path
from pathlib import Path, WindowsPath, PurePath
import torch
from torch.utils.data import DataLoader
import yaml
import cv2
import numpy as np
from easydict import EasyDict

from model import LitModule

import kafkajobs


import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_infer_transform(image_size):
    if isinstance(image_size, int):
        image_size = (image_size, image_size)    
    return A.Compose([
        # A.CenterCrop(img_size,img_size, p=1.),
        A.Resize(*image_size, interpolation=cv2.INTER_LANCZOS4),
        A.Normalize(),
        ToTensorV2(p=1.0),
    ], p=1.0)

def load_ckpt(ckpt_path, device="cuda:0"):
    if isinstance(ckpt_path, str):
        ckpt_path = Path(ckpt_path)
    with open(ckpt_path / "cfg.yml") as f:
        cfg = EasyDict(yaml.safe_load(f))
    if os.path.exists(ckpt_path / "model.ckpt"):
        model = eval(cfg['model_type']).load_from_checkpoint(ckpt_path / "model.ckpt").eval()
        print(f"load model from {ckpt_path / 'model.ckpt'}")
    else:
        raise ValueError
    return model.to(device), cfg

@torch.inference_mode()
def load_pretrained_model(ckpt, device="cuda:0"):
    model, config = load_ckpt(ckpt, device)
    print("Model loaded")
    model.eval()
    return model, config

@torch.inference_mode()
def get_embedding(model, image):    
    #  image = cv2.cvtColor(cv2.imread(self.image_path[item]), cv2.COLOR_BGR2RGB)
    # if self.transform:
    # get_infer_transform...
    #    image = self.transform(image=image)['image']
    # rst["images"]
    # batch = {k: torch.tensor(v).to(device) for k, v in data.items()}
    print(f"image shape: {image.shape}")
    
    embedding = model(image).cpu().numpy()
    #embeddings = np.vstack(embeddings)
    # embeddings = normalize(embeddings, axis=1, norm="l2")
    
    print(f"embedding size {embedding.shape}")    
    return embedding

def get_embedding_for_json(model, preproc_transform, serialized_image):
    # TODO: avoid wrapping with list
    npImage = kafkajobs.serialization.imagesFieldToNp([serialized_image])[0]
    npImage = preproc_transform(image=npImage)['image']
    # adding batch dimension
    npImage = npImage[np.newaxis, ...]
    embeddings = get_embedding(model, npImage)
    return embeddings


#def run_predict(save_dir, data_dir, model, filt=None, device='cuda:0'):   
#    get_embeddings(model, lost_query, device, lost_query_emb)


    