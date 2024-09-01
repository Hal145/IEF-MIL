from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import h5py
from utils.eval_utils import *

# Training settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='/media/nfs/final_results',
                    help='relative path to results folder, i.e. ' +
                         'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small',
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, default='clam_sb',
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--drop_out', action='store_true', default=False,
                    help='whether model uses dropout')
parser.add_argument('--scales', type=list, default=[5, 10, 20],
                    help='List of scales of patches')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False,
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping_1234'])
parser.add_argument('--no_validation', type=bool, default=False, help='disable validation')
parser.add_argument('--data_dir', type=str, default='/media/nfs/HL_dataset/clam/features/kimianet/')
parser.add_argument('--fv_len', type=int, help='Lenght of the feature vector.')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.save_dir = os.path.join('/media/nfs/eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))
print(args.models_dir)

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir,
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path='dataset_csv/tumor_vs_normal_dummy_clean.csv',
                                  data_dir=os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                                  shuffle=False,
                                  print_info=True,
                                  label_dict={'normal_tissue': 0, 'tumor_tissue': 1},
                                  patient_strat=False,
                                  ignore=[])


elif args.task == 'task_2_tumor_subtyping_1234':
    args.n_classes = 4
    dataset = Generic_MIL_Dataset(csv_path='dataset_csv/chl_subtyping.csv',
                                  data_dir=os.path.join(args.data_dir, 'chl_extracted_features_224_{}x'),
                                  list_fe=None,
                                  scales=args.scales,
                                  shuffle=False,
                                  seed=args.seed,
                                  print_info=True,
                                  label_dict={'subtype_1': 0, 'subtype_2': 1, 'subtype_3': 2, 'subtype_4': 3},
                                  patient_strat=False,
                                  ignore=[])

# elif args.task == 'tcga_kidney_cv':
#     args.n_classes=3
#     dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_kidney_clean.csv',
#                             data_dir= os.path.join(args.data_root_dir, 'tcga_kidney_20x_features'),
#                             shuffle = False,
#                             print_info = True,
#                             label_dict = {'TCGA-KICH':0, 'TCGA-KIRC':1, 'TCGA-KIRP':2},
#                             patient_strat= False,
#                             ignore=['TCGA-SARC'])

else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold + 1)
if args.no_validation==False:
    fold = 'x'
    ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold))]
    datasets_id = {'train': 0, 'test': 1, 'all': -1}
else:
    ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
    datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}


if __name__ == "__main__":
    results_dict = {}
    all_auc = []
    all_acc = []
    all_tn = []
    all_fp = []
    all_fn = []
    all_tp = []
    all_precision = []
    all_recall = []
    all_f1_score = []
    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            if args.no_validation==False:
                csv_path = '{}/splits_{}.csv'.format(args.splits_dir, 'x')
            else:
                csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(args.no_validation, from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]
        model, all_probs, all_predictions, all_labels, patient_results, test_error, auc, df, _, tn, fp, fn, tp, precision_, recall_, f1_score_ = \
            eval(split_dataset, args, ckpt_paths[ckpt_idx], folds[ckpt_idx])
        results_dict['fold_{}'.format(ckpt_idx)]=all_predictions
        results_dict['prob_{}'.format(ckpt_idx)] = all_probs
        if ckpt_idx==0:
            results_dict['labels'] = all_labels
        all_auc.append(auc)
        all_acc.append(1 - test_error)
        all_tn.append(tn)
        all_fp.append(fp)
        all_fn.append(fn)
        all_tp.append(tp)
        all_precision.append(precision_)
        all_recall.append(recall_)
        all_f1_score.append(f1_score_)
        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)

    if args.no_validation==False:
        fold='x'
        final_df = pd.DataFrame({'folds': fold, 'test_auc': all_auc, 'test_acc': all_acc, 'true_negative': all_tn,
                                 'false_positive': all_fp, 'false_negative': all_fn, 'true_positive': all_tp,
                                 'precision': all_precision, 'recall': all_recall, 'f1 score': all_f1_score})
    else:
        final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc, 'true_negative': all_tn,
                                 'false_positive': all_fp, 'false_negative': all_fn, 'true_positive': all_tp,
                                 'precision': all_precision, 'recall': all_recall, 'f1 score': all_f1_score})

    # Save other parameters to the Excel file - Hicran Aldemir
    existing_excel_file = os.path.join(args.save_dir, "confusion_matrices.xlsx")
    excel_data = pd.read_excel(existing_excel_file, sheet_name=None, engine='openpyxl')
    new_sheet_name = "Other_Metrics"

    # Create a Pandas Excel writer object to save the data to a new sheet
    with pd.ExcelWriter(existing_excel_file, engine='openpyxl') as writer:
        for sheet_name, df in excel_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        final_df.to_excel(writer, sheet_name=new_sheet_name, index=False)

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])

    acc_value, F1score, auc_value, thresholds_optimal = test(results_dict, args.save_dir)