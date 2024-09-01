import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
import sys

from transformers.image_utils import load_image

from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
import argparse
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import openslide
from torchvision import transforms

from extractor_models.resnet_custom import resnet50_baseline
import extractor_models.kimianet
# from extractor_models.ctran import ctranspath
from extractor_models.genmodel import genmodel
# from extractor_models.simclr.dsmil_simclr import dsmil_simclr

from transformers import CLIPModel, CLIPProcessor
# from models.simclr.dsmil_simclr import dsmil_simclr
from transformers import ViTModel

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked

from huggingface_hub import login
from conch.open_clip_custom import create_model_from_pretrained

login('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')

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cuda:0')


def load_image(image_path):
    """Load and preprocess image."""
    img = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust based on model requirements
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(img).unsqueeze(0)


def extract_features_from_images(image_dir, output_dir, model, model_name, batch_size=16):
    """Extract features from all images in a directory and save to .pt file."""
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.jpg') or fname.endswith('.jpeg')]
    
    # Create a DataLoader for batch processing
    dataset = [load_image(image_path) for image_path in image_paths]
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, collate_fn=lambda x: torch.cat(x, dim=0))

    model.eval()
    all_features = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            
            # Feature extraction
            if model_name == 'plip':
                features = model.get_image_features(batch)
            elif model_name == 'Conch':
                features = model.encode_image(batch, proj_contrast=False, normalize=False)
            elif model_name == 'Virchow':
                output = model(batch)
                features = output[:, 0]
            else:
                features = model(batch)

            if model_name == 'HistoSSL':
                features = features.last_hidden_state[:, 0, :]
            
            all_features.append(features.cpu())

    # Save all extracted features to a single .pt file
    all_features_tensor = torch.cat(all_features)
    torch.save(all_features_tensor, os.path.join(output_dir, 'real_features.pt'))
    print(f"Feature extraction completed! Features saved to {os.path.join(output_dir, 'extracted_features.pt')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Extraction from JPEG images')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing JPEG images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save extracted features')
    parser.add_argument('--extractor_model', type=str, default='ResNet', help='Model to use for feature extraction')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing images')
    args = parser.parse_args()

    # Load model based on extractor_model argument
    print('loading model checkpoint')
    if args.extractor_model == 'UNI':
        model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5,
                                  dynamic_img_size=True)
        model = model.to(device)
    if args.extractor_model == 'GigaPath':
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        model = model.to(device)
    if args.extractor_model == 'Conch':
        model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch")
        model = model.to(device)
    if args.extractor_model == 'Virchow':
        model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True,
                                  mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU)
        model = model.to(device)

    model = model.to(device)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Extract features and save to .pt file
    extract_features_from_images(args.image_dir, args.output_dir, model, args.extractor_model, args.batch_size)

