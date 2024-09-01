import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import shape

from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_SB, CLAM_MB, MultiScaleCLAM_MB
import pdb
import os
import pandas as pd
from itertools import cycle
from utils.utils import *
from utils.test_utils import test
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from sklearn import metrics
from scipy import interp
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from functools import partial
import models.dsmil as DSMIL
from models.transmil import TransMIL

from models.transformer import ACMIL_GA
from models.transformer import ACMIL_MHA


class Config:
    def __init__(self, D_feat, D_inner, n_class, n_token, n_masked_patch, mask_drop):
        self.D_feat = D_feat
        self.D_inner = D_inner
        self.n_class = n_class
        self.n_token = n_token
        self.n_masked_patch = n_masked_patch
        self.mask_drop = mask_drop



def initiate_model(args, ckpt_path):
    print('Init Model')
    device = 'cuda:0'
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, 'fv_len': args.fv_len}

    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})

    if args.model_type == 'clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type == 'clam_mb':
        model = CLAM_MB(**model_dict)
    elif args.model_type == 'transmil':
        model = TransMIL(n_classes=4, dim=args.fv_len).to(device)
    elif args.model_type == 'dsmil':
        i_classifier = DSMIL.FCLayer(in_size=args.fv_len, out_size=4).to(device)
        b_classifier = DSMIL.BClassifier(input_size=args.fv_len, output_class=4, dropout_v=0, nonlinear=True).to(device)
        model = DSMIL.MILNet(i_classifier, b_classifier).to(device)
    elif args.model_type == 'acmil_ga':
        if args.fv_len == 512:
            d_inner = 256
        elif args.fv_len == 1024:
            d_inner = 512
        elif args.fv_len == 768:
            d_inner = 384
        elif args.fv_len == 1536:
            d_inner = 768
        elif args.fv_len == 1280:
            d_inner = 640

        conf = Config(D_feat=args.fv_len, D_inner=d_inner, n_class=4, n_token=5, n_masked_patch=10, mask_drop=0.6)
        model = ACMIL_GA(conf, n_token=conf.n_token, n_masked_patch=conf.n_token, mask_drop=conf.mask_drop)
        model = model.to('cuda:0')

    else:  # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    print_network(model)

    ckpt = torch.load(ckpt_path, map_location=torch.device('cuda:0'))
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''): ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    # model.relocate()
    model.to(device)
    model.eval()
    return model


def eval(dataset, args, ckpt_path, fold):
    model = initiate_model(args, ckpt_path)

    print('Init Loaders')
    loader = get_simple_loader(dataset)
    all_probs, all_predictions, all_labels, patient_results, test_error, auc, df, _, _tn, fp, fn, tp, precision, recall, f1_score_ = summary(model, loader,
                                                                                                     fold,
                                                                                                     args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    return model, all_probs, all_predictions, all_labels, patient_results, test_error, auc, df, _, _tn, fp, fn, tp, precision, recall, f1_score_


def summary(model, loader, fold, args):
    device = "cuda:0"
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))


    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (data_5x, data_10x, data_20x, label, slide_id) in enumerate(loader):
        data_5x = data_5x.to(device)
        data_10x = data_10x.to(device)
        data_20x = data_20x.to(device)
        label = label.to(device)

        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            if args.model_type == 'clam_mb_ms':
                label = label.to("cuda:0")
                logits, Y_prob, Y_hat, _, instance_dict = model((data_5x, data_10x, data_20x),
                                                                label=label,
                                                                instance_eval=True)

            elif args.model_type == 'dsmil':
                label = label.to('cuda:0')
                data = torch.cat((data_5x, data_10x, data_20x), dim=0)
                ins_prediction, bag_prediction, _, _ = model(data)
                # max_prediction, _ = torch.max(ins_prediction, 0)
                # top_10 = torch.topk(ins_prediction, 10, dim=0, largest=True)[0]
                max_prediction = torch.mean(ins_prediction, dim=0)  # top_10
                Y_prob = (0.5 * torch.softmax(max_prediction.unsqueeze(0), dim=1) + 0.5 * torch.softmax(bag_prediction,
                                                                                                        dim=1))
                Y_hat = torch.argmax(Y_prob, dim=1)

            elif args.model_type == 'transmil':
                label = label.to(device)
                data = torch.cat((data_5x, data_10x, data_20x), dim=0)
                logits, Y_prob, Y_hat = model(data)

            elif args.model_type == 'clam_mb' or args.model_type == 'clam_sb':
                data = torch.cat((data_5x, data_10x, data_20x), dim=0)
                logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

            elif args.model_type == 'acmil_ga' or args.model_type == 'acmil_mha':
                # data = torch.cat((data_5x, data_10x, data_20x), dim=0)
                label = label.to(device)
                sub_preds, slide_preds, attn = model(data_20x)
                Y_prob = torch.softmax(slide_preds, dim=-1)
                Y_hat = torch.argmax(Y_prob, dim=1)

            else:
                label = label.to("cuda:0")
                logits, Y_prob, Y_hat, _, instance_dict = model([data_5x, data_10x, data_20x])

        acc_logger.log(Y_hat, label)

        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})

        error = calculate_error(Y_hat, label)
        test_error += error

    del data_5x, data_10x, data_20x
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else:
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
            print('All labels are: {}'.format(all_labels))
            print(shape(all_labels))
            print('----------------------------------------------------')
            print('All predicted labels are: {}'.format(all_preds))
            print(shape(all_preds))
            print('------------------------------------------------------')

            f1_score_all = f1_score(all_labels, all_preds, average='weighted')
            f1_score_each = f1_score(all_labels, all_preds, average=None)

            tn_, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
            print('Statistical metrics for {} class classification:'.format(args.n_classes))
            print()
            print('True Negative: {}'.format(tn_))
            print('False Pozitive: {}'.format(fp))
            print('False Negative: {}'.format(fn))
            print('True Pozitive: {}'.format(tp))
            print('F1 Score weighted: {}'.format(f1_score_all))
            print('F1 Score each: {}'.format(f1_score_each))
        else:
            class_names = ['NS', 'MC', 'LR', 'LD']  # arrange this part when new classes added
            f1_score_weighted = f1_score(all_labels, all_preds, average='weighted')

            tn_, fp, fn, tp, precision, recall = [], [], [], [], [], []
            cm = confusion_matrix(all_labels, all_preds)
            num_classes = cm.shape[0]
            print('Statistical metrics for {} class classification:'.format(num_classes))
            print('______________________________________________________________')
            epsilon = 1e-7
            for i in range(num_classes):
                tn_class = np.sum(np.delete(np.delete(cm, i, 0), i, 1))
                fp_class = np.sum(np.delete(cm, i, 0)[:, i])
                fn_class = np.sum(cm[i, :]) - cm[i, i]
                tp_class = cm[i, i]
                precision_class = tp_class / (tp_class + fp_class + epsilon)
                recall_class = tp_class / (tp_class + fn_class + epsilon)

                print('Class {} True Negative: {}'.format(i, tn_class))
                print('Class {} False Pozitive: {}'.format(i, fp_class))
                print('Class {} False Negative: {}'.format(i, fn_class))
                print('Class {} True Pozitive: {}'.format(i, tp_class))
                print('Class {} Precision: {}'.format(i, precision_class))
                print('Class {} Recall: {}'.format(i, recall_class))

                tn_.append(tn_class)
                fp.append(fp_class)
                fn.append(fn_class)
                tp.append(tp_class)
                precision.append(precision_class)
                recall.append(recall_class)

            print('F1 score weighted is: {}'.format(f1_score_weighted))
            print('Confusion Matrix is: {}'.format(cm))


            # Hicran Aldemir
            # Save confusion matrix to an excel file
            conf_matrix = pd.DataFrame(cm, columns=class_names, index=class_names)

            if fold == 0:
                cm_csv_name = os.path.join(args.save_dir, "confusion_matrices.xlsx")
                excel_writer = pd.ExcelWriter(cm_csv_name, engine='openpyxl')
                conf_matrix.to_excel(excel_writer)
                excel_writer.close()
            else:
                existing_excel_file = os.path.join(args.save_dir, "confusion_matrices.xlsx")
                excel_data = pd.read_excel(existing_excel_file, sheet_name=None)
                new_sheet_name = "fold_{}".format(fold)

                # Create a Pandas Excel writer object to save the data to a new sheet
                with pd.ExcelWriter(existing_excel_file, engine='openpyxl') as writer:
                    for sheet_name, df in excel_data.items():
                        df.to_excel(writer, sheet_name=sheet_name, index=False)

                    conf_matrix.to_excel(writer, sheet_name=new_sheet_name, index=False)
                # excel_writer.close()
            # End here

            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:, c]})
    df = pd.DataFrame(results_dict)
    return (all_probs, all_preds, all_labels, patient_results, test_error, auc_score, df, acc_logger, tn_, fp, fn, tp, precision,
            recall, f1_score_weighted)
