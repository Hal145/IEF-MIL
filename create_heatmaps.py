from __future__ import print_function

import numpy as np
import argparse
import torch
import torch.nn as nn
# import pdby
import os
import pandas as pd
from utils.utils import *
from math import floor
from utils.eval_utils import initiate_model as initiate_model
from models.model_clam import CLAM_MB, CLAM_SB, MultiScaleCLAM_MB
from models.dsmil import MILNet
import h5py
import yaml
from wsi_core.batch_process_utils import initialize_df
from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches
from wsi_core.wsi_utils import sample_rois
from utils.file_utils import save_hdf5

from huggingface_hub import login
from conch.open_clip_custom import create_model_from_pretrained
login('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')

parser = argparse.ArgumentParser(description='Heatmap inference script')
parser.add_argument('--save_exp_code', type=str, default=None,
					help='experiment code')
parser.add_argument('--overlap', type=float, default=None)
parser.add_argument('--config_file', type=str, default="heatmap_config_template.yaml")
args = parser.parse_args()

def infer_single_slide(model, features, label, reverse_label_dict, len_feats, k=1):
	device = torch.device('cuda:0')
	data_5x, data_10x, data_20x = features
	data_5x, data_10x, data_20x = data_5x.to(device), data_10x.to(device), data_20x.to(device)
	with torch.no_grad():
		if isinstance(model, (CLAM_SB, CLAM_MB)):
			# model_results_dict = model(features)
			logits, Y_prob, Y_hat, A, _ = model(data_5x, data_10x, data_20x)
			# logits, Y_prob, Y_hat, A, _ = model((x1, x2, x3), label=label, instance_eval=False)
			Y_hat = Y_hat.item()

			if isinstance(model, (CLAM_MB, MultiScaleCLAM_MB)):
				A = [A__[Y_hat] for A__ in A]

			A = [A_.view(-1, 1).cpu().numpy()for A_ in A]

		elif isinstance(model, MILNet):
			data = torch.cat((data_5x, data_10x, data_20x), dim=0)
			ins_prediction, bag_prediction, A, _ = model(data)
			max_prediction = torch.mean(ins_prediction, dim=0)  # top_10
			Y_prob = (0.2 * torch.softmax(max_prediction.unsqueeze(0), dim=1) + 0.8 * torch.softmax(bag_prediction, dim=1))
			Y_hat = torch.argmax(Y_prob, dim=1).item()
			A_ = A[:, Y_hat]
			A_ = A_.view(-1, 1).cpu().numpy()

			split_points = np.cumsum(len_feats[:-1])
			A_final = np.split(A_, split_points, axis=0)
		else:
			raise NotImplementedError
		
		probs, ids = torch.topk(Y_prob, k)
		probs = probs[-1].cpu().numpy()
		ids = ids[-1].cpu().numpy()
		preds_str = np.array([reverse_label_dict[idx] for idx in ids])

	return ids, preds_str, probs, A_final


def load_params(df_entry, params):
	for key in params.keys():
		if key in df_entry.index:
			dtype = type(params[key])
			val = df_entry[key] 
			val = dtype(val)
			if isinstance(val, str):
				if len(val) > 0:
					params[key] = val
			elif not np.isnan(val):
				params[key] = val
			else:
				pdb.set_trace()

	return params

def parse_config_dict(args, config_dict):
	if args.save_exp_code is not None:
		config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
	if args.overlap is not None:
		config_dict['patching_arguments']['overlap'] = args.overlap
	return config_dict


def split_image(image, patch_size=2048, i=0, slide_idd='', output_dir="patches"):
	width, height = image.size

	# Create the output directory if it doesn't exist
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	patch_count = 0
	# Loop over the image and create patches
	for x in range(0, width, patch_size):
		for y in range(0, height, patch_size):
			# Define the box to extract
			box = (x, y, x + patch_size, y + patch_size)
			# Extract the patch
			patch = image.crop(box)
			# Save the patch
			patch_file = os.path.join(output_dir, f"{i}_{slide_idd}_{patch_count}.png")
			patch.save(patch_file)
			patch_count += 1

	print(f"Total patches saved: {patch_count}")


if __name__ == '__main__':
	config_path = os.path.join('heatmaps/configs', args.config_file)
	config_dict = yaml.safe_load(open(config_path, 'r'))
	config_dict = parse_config_dict(args, config_dict)

	for key, value in config_dict.items():
		if isinstance(value, dict):
			print('\n'+key)
			for value_key, value_value in value.items():
				print (value_key + " : " + str(value_value))
		else:
			print ('\n'+key + " : " + str(value))
			
	"""decision = input('Continue? Y/N ')
	if decision in ['Y', 'y', 'Yes', 'yes']:
		pass
	elif decision in ['N', 'n', 'No', 'NO']:
		exit()
	else:
		raise NotImplementedError"""

	args = config_dict
	patch_args = argparse.Namespace(**args['patching_arguments'])
	data_args = argparse.Namespace(**args['data_arguments'])
	model_args = args['model_arguments']
	model_args.update({'n_classes': args['exp_arguments']['n_classes'], 'fv_len': 512})
	model_args = argparse.Namespace(**model_args)
	exp_args = argparse.Namespace(**args['exp_arguments'])
	heatmap_args = argparse.Namespace(**args['heatmap_arguments'])
	sample_args = argparse.Namespace(**args['sample_arguments'])

	patch_size = tuple([patch_args.patch_size for i in range(2)])
	step_size = tuple((np.array(patch_size) * (1-patch_args.overlap)).astype(int))
	print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(patch_size[0], patch_size[1], patch_args.overlap, step_size[0], step_size[1]))

	
	preset = data_args.preset
	def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False, 
					  'keep_ids': 'none', 'exclude_ids':'none'}
	def_filter_params = {'a_t':50.0, 'a_h': 8.0, 'max_n_holes':10}
	def_vis_params = {'vis_level': -1, 'line_thickness': 250}
	def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

	if preset is not None:
		preset_df = pd.read_csv(preset)
		for key in def_seg_params.keys():
			def_seg_params[key] = preset_df.loc[0, key]

		for key in def_filter_params.keys():
			def_filter_params[key] = preset_df.loc[0, key]

		for key in def_vis_params.keys():
			def_vis_params[key] = preset_df.loc[0, key]

		for key in def_patch_params.keys():
			def_patch_params[key] = preset_df.loc[0, key]


	if data_args.process_list is None:
		if isinstance(data_args.data_dir, list):
			slides = []
			for data_dir in data_args.data_dir:
				slides.extend(os.listdir(data_dir))
		else:
			slides = sorted(os.listdir(data_args.data_dir))
		slides = [slide for slide in slides if data_args.slide_ext in slide]
		df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)
		
	else:
		df = pd.read_csv(os.path.join('heatmaps/process_lists', data_args.process_list))
		df = initialize_df(df, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)

	mask = df['process'] == 1
	process_stack = df[mask].reset_index(drop=True)
	total = len(process_stack)
	print('\nlist of slides to process: ')
	print(process_stack.head(len(process_stack)))

	print('\ninitializing model from checkpoint')
	ckpt_path = model_args.ckpt_path
	print('\nckpt path: {}'.format(ckpt_path))
	
	if model_args.initiate_fn == 'initiate_model':
		model = initiate_model(model_args, ckpt_path)
	else:
		raise NotImplementedError

	feature_extractor, preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch")
	# feature_extractor = resnet50_baseline(pretrained=True)
	feature_extractor.eval()

	device= torch.device('cuda:0') # torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print('Done!')

	label_dict =  data_args.label_dict
	class_labels = list(label_dict.keys())
	class_encodings = list(label_dict.values())
	reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))} 

	# if torch.cuda.device_count() > 1:
	# 	device_ids = list(range(torch.cuda.device_count()))
	# 	feature_extractor = nn.DataParallel(feature_extractor, device_ids=device_ids).to('cuda:0')
	# else:
	# 	feature_extractor = feature_extractor.to(device)
	feature_extractor = feature_extractor.to(device)

	os.makedirs(exp_args.production_save_dir, exist_ok=True)
	os.makedirs(exp_args.raw_save_dir, exist_ok=True)
	blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': patch_size, 
	'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}

	for i in range(len(process_stack)):
		slide_name = process_stack.loc[i, 'slide_id']
		if data_args.slide_ext not in slide_name:
			slide_name+=data_args.slide_ext
		print('\nprocessing: ', slide_name)	

		try:
			label = process_stack.loc[i, 'label']
		except KeyError:
			label = 'Unspecified'

		slide_id = slide_name.replace(data_args.slide_ext, '')

		if not isinstance(label, str):
			grouping = reverse_label_dict[label]
		else:
			grouping = label

		p_slide_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, str(grouping))
		os.makedirs(p_slide_save_dir, exist_ok=True)

		r_slide_save_dir = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, str(grouping),  slide_id)
		os.makedirs(r_slide_save_dir, exist_ok=True)

		if heatmap_args.use_roi:
			x1, x2 = process_stack.loc[i, 'x1'], process_stack.loc[i, 'x2']
			y1, y2 = process_stack.loc[i, 'y1'], process_stack.loc[i, 'y2']
			top_left = (int(x1), int(y1))
			bot_right = (int(x2), int(y2))
		else:
			top_left = None
			bot_right = None
		
		print('slide id: ', slide_id)
		print('top left: ', top_left, ' bot right: ', bot_right)

		if isinstance(data_args.data_dir, str):
			slide_path = os.path.join(data_args.data_dir, slide_name)
		elif isinstance(data_args.data_dir, dict):
			data_dir_key = process_stack.loc[i, data_args.data_dir_key]
			slide_path = os.path.join(data_args.data_dir[data_dir_key], slide_name)
		else:
			raise NotImplementedError

		mask_file = os.path.join(r_slide_save_dir, slide_id+'_mask.pkl')
		
		# Load segmentation and filter parameters
		seg_params = def_seg_params.copy()
		filter_params = def_filter_params.copy()
		vis_params = def_vis_params.copy()

		seg_params = load_params(process_stack.loc[i], seg_params)
		filter_params = load_params(process_stack.loc[i], filter_params)
		vis_params = load_params(process_stack.loc[i], vis_params)

		keep_ids = str(seg_params['keep_ids'])
		if len(keep_ids) > 0 and keep_ids != 'none':
			seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
		else:
			seg_params['keep_ids'] = []

		exclude_ids = str(seg_params['exclude_ids'])
		if len(exclude_ids) > 0 and exclude_ids != 'none':
			seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
		else:
			seg_params['exclude_ids'] = []

		for key, val in seg_params.items():
			print('{}: {}'.format(key, val))

		for key, val in filter_params.items():
			print('{}: {}'.format(key, val))

		for key, val in vis_params.items():
			print('{}: {}'.format(key, val))
		
		print('Initializing WSI object')
		wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
		print('Done!')

		wsi_ref_downsample = [wsi_object.level_downsamples[i] for i in patch_args.patch_level]

		# the actual patch size for heatmap visualization should be the patch size * downsample factor * custom downsample factor
		vis_patch_size = [tuple((np.array(patch_size) * np.array(ds) * patch_args.custom_downsample).astype(int))
						  for ds in wsi_ref_downsample]

		block_map_save_path = [os.path.join(r_slide_save_dir, '{}_{}x_blockmap.h5'.format(slide_id, scale_)) for scale_ in data_args.scale]
		mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))
		if vis_params['vis_level'] < 0:
			best_level = wsi_object.wsi.get_best_level_for_downsample(32)
			vis_params['vis_level'] = best_level
		mask = wsi_object.visWSI(**vis_params, number_contours=True)
		mask.save(mask_path)

		data_path = [os.path.join(data_args.processed_dir, 'chl_extracted_features_224_{}x'.format(scale)) for scale in data_args.scale]
		features_path = [os.path.join(_path, 'pt_files', slide_id+'.pt') for _path in data_path]
		h5_path = [os.path.join(_path, 'h5_files', slide_id+'.h5') for _path in data_path]
	

		##### check if h5_features_file exists ######
		"""if not os.path.isfile(h5_path) :
			_, _, wsi_object = compute_from_patches(wsi_object=wsi_object, 
											model=model, 
											feature_extractor=feature_extractor, 
											batch_size=exp_args.batch_size, **blocky_wsi_kwargs, 
											attn_save_path=None, feat_save_path=h5_path, 
											ref_scores=None)	"""
		
		##### check if pt_features_file exists ######
		"""if not os.path.isfile(features_path):
			file = h5py.File(h5_path, "r")
			features = torch.tensor(file['features'][:])
			torch.save(features, features_path)
			file.close()"""

		# load features
		features = [torch.load(_path) for _path in features_path]
		# features = [feature[~torch.isnan(feature).any(dim=1)] for feature in features] # remove nan values in features if there is any
		features = [torch.nan_to_num(feature, nan=0) for feature in features]
		len_feats = [len(feature) for feature in features]
		process_stack.loc[i, 'bag_size1'] = len_feats[0]
		process_stack.loc[i, 'bag_size2'] = len_feats[1]
		process_stack.loc[i, 'bag_size3'] = len_feats[2]
		
		wsi_object.saveSegmentation(mask_file)
		Y_hats, Y_hats_str, Y_probs, A = infer_single_slide(model, features, label, reverse_label_dict, len_feats,
															exp_args.n_classes)
		del features

		block_map_save_paths = []
		if not os.path.isfile(block_map_save_path[0]):
			for i, A_ in enumerate(A):
				file = h5py.File(h5_path[i], "r")
				coords = file['coords'][:]
				file.close()
				asset_dict = {'attention_scores': A_, 'coords': coords}
				block_map_save_path_ = save_hdf5(block_map_save_path[i], asset_dict, mode='w')
				block_map_save_paths.append(block_map_save_path_)
		
		# save top 3 predictions
		for c in range(exp_args.n_classes):
			process_stack.loc[i, 'Pred_{}'.format(c)] = Y_hats_str[c]
			process_stack.loc[i, 'p_{}'.format(c)] = Y_probs[c]

		os.makedirs('heatmaps/results/', exist_ok=True)
		if data_args.process_list is not None:
			process_stack.to_csv('heatmaps/results/{}.csv'.format(data_args.process_list.replace('.csv', '')), index=False)
		else:
			process_stack.to_csv('heatmaps/results/{}.csv'.format(exp_args.save_exp_code), index=False)
		
		scores, coords = [], []
		for h5_save_path in block_map_save_path:
			file = h5py.File(h5_save_path, 'r')
			dset = file['attention_scores']
			coord_dset = file['coords']
			score = dset[:]
			coord = coord_dset[:]
			scores.append(score)
			coords.append(coord)
			file.close()

		samples = sample_args.samples
		for sample in samples:
			if sample['sample']:
				for i, (score_scale, coord_scale) in enumerate(zip(scores, coords)):
					tag = "label_{}_pred_{}".format(label, Y_hats[0])
					sample_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code,
												   'sampled_patches', str(tag), sample['name'])
					os.makedirs(sample_save_dir, exist_ok=True)
					print('sampling {}'.format(sample['name']))
					sample_results = sample_rois(score_scale, coord_scale, k=sample['k'], mode=sample['mode'],
												 seed=sample['seed'],
												 score_start=sample.get('score_start', 0),
												 score_end=sample.get('score_end', 1))
					for idx, (s_coord, s_score) in enumerate(
							zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
						print('coord: {} score: {:.3f}'.format(s_coord, s_score))
						patch = wsi_object.wsi.read_region(tuple(s_coord), patch_args.patch_level[i],
														   (patch_args.patch_size, patch_args.patch_size)).convert(
							'RGB')
						patch.save(os.path.join(sample_save_dir,
												'{}__{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(i, idx, slide_id, s_coord[0],
																					  s_coord[1], s_score)))

		wsi_kwargs = {'top_left': top_left, 'bot_right': bot_right, 'patch_size': patch_size, 'step_size': step_size,
		'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}

		heatmap_save_name = ['{}__{}_blockmap.tiff'.format(slide_id, level) for level in wsi_kwargs['level']]
		if os.path.isfile(os.path.join(r_slide_save_dir, heatmap_save_name[0])):
			pass
		else:
			for i,(_score, _coord) in enumerate(zip(scores, coords)):
				heatmap_ = drawHeatmap(_score, _coord, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap,
						    alpha=heatmap_args.alpha, use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
							thresh=-1, patch_size = vis_patch_size[i], convert_to_percentiles=True)

				heatmap_.save(os.path.join(r_slide_save_dir, '{}__{}_blockmap.png'.format(i, slide_id)))
				# split_image(heatmap_, patch_size=2048, i=i, slide_idd=slide_id, output_dir=r_slide_save_dir)
			del heatmap_

		save_path = os.path.join(r_slide_save_dir, '{}_{}_roi_{}.h5'.format(slide_id, patch_args.overlap, heatmap_args.use_roi))

		if heatmap_args.use_ref_scores:
			ref_scores = scores
		else:
			ref_scores = None
		
		if heatmap_args.calc_heatmap:
			compute_from_patches(wsi_object=wsi_object, clam_pred=Y_hats[0], model=model, feature_extractor=feature_extractor, batch_size=exp_args.batch_size, **wsi_kwargs, 
								attn_save_path=save_path,  ref_scores=ref_scores)

		if not os.path.isfile(save_path):
			print('heatmap {} not found'.format(save_path))
			if heatmap_args.use_roi:
				save_path_full = os.path.join(r_slide_save_dir, '{}_{}_roi_False.h5'.format(slide_id, patch_args.overlap))
				print('found heatmap for whole slide')
				save_path = save_path_full
			else:
				continue

		file = h5py.File(save_path, 'r')
		dset = file['attention_scores']
		coord_dset = file['coords']
		scores = dset[:]
		coords = coord_dset[:]
		file.close()

		heatmap_vis_args = {'convert_to_percentiles': True, 'vis_level': heatmap_args.vis_level, 'blur': heatmap_args.blur, 'custom_downsample': heatmap_args.custom_downsample}
		if heatmap_args.use_ref_scores:
			heatmap_vis_args['convert_to_percentiles'] = False

		heatmap_save_name = '{}_{}_roi_{}_blur_{}_rs_{}_bc_{}_a_{}_l_{}_bi_{}_{}.{}'.format(slide_id, float(patch_args.overlap), int(heatmap_args.use_roi),
																						int(heatmap_args.blur), 
																						int(heatmap_args.use_ref_scores), int(heatmap_args.blank_canvas), 
																						float(heatmap_args.alpha), int(heatmap_args.vis_level), 
																						int(heatmap_args.binarize), float(heatmap_args.binary_thresh), heatmap_args.save_ext)


		if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
			pass
		
		else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
			heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object,  
						          cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, **heatmap_vis_args, 
						          binarize=heatmap_args.binarize, 
						  		  blank_canvas=heatmap_args.blank_canvas,
						  		  thresh=heatmap_args.binary_thresh,  patch_size = vis_patch_size,
						  		  overlap=patch_args.overlap, 
						  		  top_left=top_left, bot_right = bot_right)
			if heatmap_args.save_ext == 'jpg':
				heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
			else:
				heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))
		
		if heatmap_args.save_orig:
			if heatmap_args.vis_level >= 0:
				vis_level = heatmap_args.vis_level
			else:
				vis_level = vis_params['vis_level']
			heatmap_save_name = '{}_orig_{}.{}'.format(slide_id,int(vis_level), heatmap_args.save_ext)
			if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
				pass
			else:
				heatmap = wsi_object.visWSI(vis_level=vis_level, view_slide_only=True, custom_downsample=heatmap_args.custom_downsample)
				if heatmap_args.save_ext == 'jpg':
					heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
				else:
					heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))

	with open(os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, 'config.yaml'), 'w') as outfile:
		yaml.dump(config_dict, outfile, default_flow_style=False)


