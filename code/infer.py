# The code is adopted frin Zirui's code (https://raw.githubusercontent.com/LostPetInitiative/study_spring_2022/main/zhirui/infer.py)

import os.path
from pathlib import Path, WindowsPath, PurePath
import torch
from torch.utils.data import DataLoader
import yaml
import cv2
import numpy as np
from easydict import EasyDict

import copy

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
    #print(f"image shape: {image.shape}")
    
    embedding = model(image).cpu().numpy()
    
    #print(f"embedding size {embedding.shape}")    
    return embedding

def get_embedding_for_json(model, preproc_transform, serialized_image):
    # TODO: avoid wrapping with list
    npImage = kafkajobs.serialization.imagesFieldToNp([serialized_image])[0]
    npImage = preproc_transform(image=npImage)['image']
    
    # adding batch dimension
    npImage = npImage[np.newaxis, ...]
    embeddings = get_embedding(model, npImage)
    return embeddings

def process_job(model, preproc_transform, job):
    output_job = copy.deepcopy(job)
    yolo5_output = output_job["yolo5_output"]
    del output_job["yolo5_output"]

    for entry in yolo5_output:        
        entry["embedding"] = kafkajobs.serialization.npArrayToBase64str(get_embedding_for_json(model, preproc_transform, entry["head"]))
        del entry["head"]
        del entry["annotated"]
        
    output_job["image_embeddings"] = yolo5_output
    

    return output_job

        

    