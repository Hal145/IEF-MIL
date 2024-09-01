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


def compute_w_loader(file_path, output_path, wsi, model, model_name,
					 batch_size=8, verbose=0, print_every=20, pretrained=True,
					 custom_downsample=1, target_patch_size=-1):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained,
								 custom_downsample=custom_downsample, target_patch_size=target_patch_size)
	x, y = dataset[0]
	kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path, len(loader)))

	mode = 'w'
	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))

			batch = batch.to(device, non_blocking=True)
			ccc = torch.isnan(batch).any()
			print('count {}, {}'.format(count, ccc))
			if ccc:
				# with open('test.npy', 'wb') as f:
					# np.save(f, batch.cpu().numpy())
				print("Condition met, stopping code execution.")
				sys.exit(0)

			if model_name == 'plip':
				features = model.get_image_features(batch)
			elif model_name == 'Conch':
				with torch.inference_mode():
					features = model.encode_image(batch, proj_contrast=False, normalize=False)
			elif model_name == 'Virchow':
				output = model(batch)
				features = output[:, 0]  # size: 1 x 1280
				# patch_tokens = output[:, 1:]  # size: 1 x 256 x 1280

				# concatenate class token and average pool of patch tokens
				# features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
			else:
				features = model(batch)

			if model_name == 'HistoSSL':
				features = features.last_hidden_state[:, 0, :]

			ddd = torch.isnan(features).any()
			print('count {}, {}'.format(count, ddd))
			features = features.cpu().numpy()

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
			mode = 'a'

	return output_path

def load_image(image_path):
	with Image.open(image_path) as img:
		# Preprocess the image here if needed (e.g., resize, normalization)
		resized_img = img.resize((2048, 2048), Image.ANTIALIAS)
	return np.array(resized_img)


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--extractor_model', type=str, default='ResNet')
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.ndpi')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
args = parser.parse_args()


if __name__ == '__main__':

	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)

	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	print('loading model checkpoint')
	if args.extractor_model == 'KimiaNet':
		model = extractor_models.kimianet.KimiaNet(imagenet=False)
		model = model.to(device)
	if args.extractor_model == 'ResNet':
		model = resnet50_baseline(pretrained=True)
		model = model.to(device)
	r"""if args.extractor_model == 'CTransPath':
		model = ctranspath()
		model.head = nn.Identity()
		td = torch.load('./saved_models/ctranspath.pth')
		model.load_state_dict(td['model'], strict=True)
		model = model.to(device)"""
	if args.extractor_model == 'BROW':
		model = genmodel(ckpt='./saved_models/vitb_wsi_proposed.pth')
		model = model.to(device)
	if args.extractor_model == 'HistoSSL':
		model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
		model = model.to(device)
	if args.extractor_model == 'plip':
		model = CLIPModel.from_pretrained('vinid/plip', use_auth_token=None)
		model = model.to(device)
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


	# print_network(model)
	# if torch.cuda.device_count() > 1:
	# 	model = nn.DataParallel(model)

	model.eval()
	total = len(bags_dataset)

	for bag_candidate_idx in range(total):
		slide_id, slide_ex = bags_dataset[bag_candidate_idx].split('.')
		args.slide_ext = '.' + slide_ex
		bag_name = slide_id + '.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		if not args.no_auto_skip and slide_id + '.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue

		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		time_start = time.time()
		wsi = openslide.open_slide(slide_file_path)

		output_file_path = compute_w_loader(h5_file_path, output_path, wsi,
											model=model, model_name=args.extractor_model,
											batch_size=args.batch_size, verbose=1, print_every=20,
											custom_downsample=args.custom_downsample,
											target_patch_size=args.target_patch_size)

		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
		file = h5py.File(output_file_path, "r")

		features = file['features'][:]
		print('features size: ', features.shape)
		print('coordinates size: ', file['coords'].shape)
		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base + '.pt'))

		print('Feature extraction completed!')





