import pdb
import os
import pandas as pd
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str,
                    choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping_123', 'task_2_tumor_subtyping_12',
                             'task_2_tumor_subtyping_1234'])
parser.add_argument('--scales', type=list, default=[5, 10, 20],
                    help='List of scales of patches')
parser.add_argument('--val_frac', type=float, default=0.05,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default=0.05,
                    help='fraction of labels for test (default: 0.1)')
parser.add_argument('--no_validation', action='store_true', default=False, help='disable validation')
parser.add_argument('--list_fe', type=str, default=['ctranspath', 'histossl'], help="list of feature "
                                                                                    "extraction methods for fusion")

args = parser.parse_args()

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes = 2
    dataset = Generic_WSI_Classification_Dataset(csv_path='dataset_csv/TCGA_survival_patients.csv',
                                                 # !! /truba/home/haldemir/Projects/CLAM_Project/CLAM/
                                                 shuffle=False,
                                                 seed=args.seed,
                                                 print_info=True,
                                                 label_dict={'Alive': 0, 'Dead': 1},
                                                 patient_strat=True,
                                                 ignore=[])

elif args.task == 'task_2_tumor_subtyping_123':
    args.n_classes = 3
    dataset = Generic_MIL_Dataset(csv_path='dataset_csv/chl_subtyping_123.csv',
                                  data_dir='/media/nfs/HL_dataset/chl_extracted_features/',
                                  shuffle=False,
                                  seed=args.seed,
                                  print_info=True,
                                  label_dict={'subtype_1': 0, 'subtype_2': 1, 'subtype_3': 2},
                                  patient_strat=False,
                                  ignore=[])
elif args.task == 'task_2_tumor_subtyping_12':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path='dataset_csv/chl_subtyping_12.csv',
                                  data_dir=' /media/nfs/HL_dataset/chl_extracted_features/',
                                  shuffle=False,
                                  seed=args.seed,
                                  print_info=True,
                                  label_dict={'subtype_1': 0, 'subtype_2': 1},
                                  patient_strat=False,
                                  ignore=[])
elif args.task == 'chl_tumor_subtyping':
    args.n_classes = 4
    dataset = Generic_MIL_Dataset(csv_path='dataset_csv/chl_subtyping.csv',
                                  data_dir='/media/nfs/HL_dataset/clam/features/ctranspath/chl_extracted_features_224_{}x_ctranspath',
                                  list_fe=args.list_fe,
                                  shuffle=False,
                                  scales=args.scales,
                                  seed=args.seed,
                                  print_info=True,
                                  label_dict={'subtype_1': 0, 'subtype_2': 1, 'subtype_3': 2, 'subtype_4': 3},
                                  patient_strat=False,
                                  ignore=[])

else:
    raise NotImplementedError

num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)

if __name__ == '__main__':
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]

    for lf in label_fracs:
        split_dir = 'splits/' + str(args.task) + '_{}'.format(int(lf * 100))
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k=args.k, val_num=val_num, test_num=test_num, label_frac=lf)
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(args.no_validation, from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)),
                        boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))


