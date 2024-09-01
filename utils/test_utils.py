import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay, precision_recall_fscore_support
from scipy import interp
from sklearn.preprocessing import label_binarize
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict


def test(results_dict, save_dir):
    num_classes = 4

    # compute accuracy and F1score
    F1score = dict()

    result_path = os.path.join(save_dir, 'inference')
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    all_acc = []
    for i in range(5):
        label_array = results_dict['labels']
        prediction_array = results_dict['fold_{}'.format(i)]
        acc_value = accuracy_score(label_array, prediction_array)
        all_acc.append(acc_value)
    avg_acc = sum(all_acc) / len(all_acc)
    print('Average accuracy is', avg_acc)

    classes = ['NSCHL', 'MCCHL', 'LRCHL', 'LDCHL']
    all_cm = []
    for i in range(5):
        label_ndarray = results_dict['labels']
        prediction_ndarray = results_dict['fold_{}'.format(i)]
        cm = confusion_matrix(label_ndarray, prediction_ndarray)
        all_cm.append(cm)
    avg_cm = (all_cm[0] + all_cm[1] + all_cm[2] + all_cm[3] + all_cm[
        4]) / 5  # average confusion matrix for 5-fold cross-validation
    avg_cm = np.around(avg_cm)
    cm_save_name = os.path.join(save_dir, 'inference/cm.png')
    title = 'Confusion matrix of CHL classification'
    disp_confusion_matrix(avg_cm, cm_save_name, classes=classes, title=title)

    f1_macro_all, f1_micro_all = [], []
    for i in range(5):
        label_array = results_dict['labels']
        prediction_array = results_dict['fold_{}'.format(i)]
        f1_macro_all.append(f1_score(label_array, prediction_array, average='macro'))
        f1_micro_all.append(f1_score(label_array, prediction_array, average='micro'))
    avg_f1_macro = sum(f1_macro_all) / len(f1_macro_all)
    avg_f1_micro = sum(f1_micro_all) / len(f1_micro_all)
    F1score["macro"] = avg_f1_macro
    F1score["micro"] = avg_f1_micro

    auc_values = []
    for i in range(5):  # for each fold of cross validation
        _labels = results_dict['labels']
        _probs = results_dict['prob_{}'.format(i)]
        auc_value, fprs, tprs, roc_auc, _, thresholds_optimal = multi_label_roc(i, _labels, _probs, num_classes,
                                                                                save_dir)
        multi_label_mean_roc(results_dict, num_classes, save_dir)
        fpr_macro = fprs["macro"]
        tpr_macro = tprs["macro"]
        roc_auc_macro = roc_auc["macro"]
        np.save(os.path.join(save_dir, 'inference/fpr_macro_fold_{}.npy'.format(i)), fpr_macro)
        np.save(os.path.join(save_dir, 'inference/tpr_macro_fold_{}.npy'.format(i)), tpr_macro)
        np.save(os.path.join(save_dir, 'inference/roc_auc_macro_fold_{}.npy'.format(i)), roc_auc_macro)

    auc_values.append(auc_value)

    for i in range(5):
        labels_ = results_dict['labels']
        probs_ = results_dict['prob_{}'.format(i)]

        pr_curves = get_precision_recall(labels_, probs_)
        display_PRcurve(i, pr_curves, 4, save_dir)
        get_average_precision_recall(results_dict, save_dir)
        recall_macro = pr_curves["macro"]["recall"]
        precision_macro = pr_curves["macro"]["precision"]
        pr_auc_macro = pr_curves["macro"]["average_precision"]
        np.save(os.path.join(save_dir, 'inference/recall_macro_fold_{}.npy'.format(i)), recall_macro)
        np.save(os.path.join(save_dir, 'inference/precision_macro_fold_{}.npy'.format(i)), precision_macro)
        np.save(os.path.join(save_dir, 'inference/pr_auc_macro_fold_{}.npy'.format(i)), pr_auc_macro)

    return avg_acc, F1score, auc_values, thresholds_optimal


def disp_confusion_matrix(cm, save_name, classes, title):
    plt.rcParams.update({'font.size': 14})
    disp = ConfusionMatrixDisplay(cm, display_labels=np.array(classes))
    disp.plot(cmap='PuRd')
    # plt.title(title)
    plt.savefig(save_name, format='png')
    # plt.show()


def multi_label_mean_roc(results_dict, num_classes, save_dir):
    mean_fpr = np.linspace(0, 1, 100)

    fprs = dict()
    tprs = dict()
    roc_auc = dict()
    for fold in range(5):
        labels = results_dict['labels']
        binary_labels = label_binarize(labels, classes=[i for i in range(num_classes)])  # binarize labels
        fold_probs = results_dict['prob_{}'.format(fold)]  # take probs for fold _

        for c in range(0, num_classes):
            label = binary_labels[:, c]
            class_prob = fold_probs[:, c]
            # fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
            fpr, tpr, _ = roc_curve(label, class_prob)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0

            fprs['fold_{}_class_{}'.format(fold, c)], tprs['fold_{}_class_{}'.format(fold, c)] = (
                mean_fpr, interp_tpr)
            roc_auc['fold_{}_class_{}'.format(fold, c)] = auc(fprs['fold_{}_class_{}'.format(fold, c)],
                                                              tprs['fold_{}_class_{}'.format(fold, c)])

    mean_tprs = {}
    mean_aucs = {}
    for cls in range(0, num_classes):
        class_tprs = [tprs['fold_{}_class_{}'.format(f, cls)] for f in range(3)]
        mean_tpr = np.mean(class_tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)

        mean_tprs[cls] = mean_tpr
        mean_aucs[cls] = mean_auc

    # Plot all ROC curves
    lw = 2
    plt.figure(figsize=(6, 6))

    cls = {0: 'NSCHL', 1: 'MCCHL', 2: 'LRCHL', 3: 'LDCHL'}
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'magenta'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(mean_fpr, mean_tprs[i], color=color, lw=lw,
                 label = '{}  {:.3f}'.format(cls[i], mean_aucs[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('FPR', fontsize=16)
    plt.ylabel('TPR', fontsize=16)
    # plt.title('Receiver Operating Characteristic', fontsize=16)
    plt.legend(loc="lower right", fontsize=16)

    # Customizing the plot to only show 0 and 1 on the axes
    plt.xticks([0, 1], fontsize=12)
    plt.yticks([0, 1], fontsize=12)
    plt.grid(which='major', linestyle='--', linewidth='0.5', color='gray')
    plt.savefig(os.path.join(save_dir, 'inference/average_ROC.png'))
    # plt.show()


def multi_label_roc(fold, labels, probs, num_classes, save_dir):
    binary_labels = label_binarize(labels, classes=[i for i in range(num_classes)])
    # binary_predictions = label_binarize(predictions, classes=[i for i in range(num_classes)])

    fprs = dict()
    tprs = dict()
    roc_auc = dict()

    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(probs.shape) == 1:
        predictions = probs[:, None]
    for c in range(0, num_classes):
        label = binary_labels[:, c]
        class_prob = probs[:, c]
        # fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr, tpr, threshold = roc_curve(label, class_prob)

        fprs[c], tprs[c] = fpr, tpr
        roc_auc[c] = auc(fprs[c], tprs[c])

        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, class_prob)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)

    # --------------------------------------------------------------------------------------------------------------#
    # Compute micro-average ROC curve and ROC area（method1）
    # fprs["micro"], tprs["micro"], _ = roc_curve(binary_labels, binary_predictions)
    #  roc_auc["micro"] = auc(fprs["micro"], tprs["micro"])

    # Compute macro-average ROC curve and ROC area（method2）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fprs[i] for i in range(num_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fprs[i], tprs[i])
    # Finally average it and compute AUC
    mean_tpr /= num_classes
    fprs["macro"] = all_fpr
    tprs["macro"] = mean_tpr
    roc_auc["macro"] = auc(fprs["macro"], tprs["macro"])
    print("auc_macro:", roc_auc["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure(dpi=1000)
    plt.rcParams.update({'font.size': 14})
    r"""plt.plot(fprs["macro"], tprs["macro"],
             label='macro-average ROC curve (area = {0:0.3f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)"""

    cls = {0: 'NSCHL', 1: 'MCCHL', 2: 'LRCHL', 3: 'LDCHL'}
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'magenta'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fprs[i], tprs[i], color=color, lw=lw,
                 label='{}  {:.3f}'.format(cls[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # plt.xticks(fontsize=13)
    # plt.yticks(fontsize=13)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('The Receiver Operating Characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, 'inference/fold_{}_ROC.png'.format(fold)))
    # plt.show()
    # Customizing the plot to only show 0 and 1 on the axes
    # --------------------------------------------------------------------------------------------------------------#

    return aucs, fprs, tprs, roc_auc, thresholds, thresholds_optimal


def get_average_precision_recall(results_dict, save_dir):
    n_classes = 4
    mean_recall = np.linspace(0, 1, 100)
    pr_curves = {}
    # Compute PR curve and average precision for each class
    for fold in range(5):
        y_true = results_dict['labels']
        y_binary = label_binarize(y_true, classes=[i for i in range(n_classes)])
        y_score = results_dict['prob_{}'.format(fold)]
        for i in range(n_classes):
            precision, recall, _ = metrics.precision_recall_curve(y_binary[:, i], y_score[:, i])
            # precision = np.delete(precision, -1)
            # recall = np.delete(recall, -1)

            # interp_precision = np.interp(mean_recall, recall, precision)
            pr_curves['fold_{}_class_{}'.format(fold, i)] = {"precision": precision, "recall": recall,
                                                             "average_precision": metrics.average_precision_score(
                                                                 y_binary[:, i], y_score[:, i])}

    mean_precisions = {}
    mean_scores = {}
    mean_recalls = {}
    for cls in range(0, n_classes):
        class_precision = [pr_curves['fold_{}_class_{}'.format(f, cls)]['precision'] for f in range(3)]
        mean_precision = np.mean(class_precision, axis=0)
        class_recall = [pr_curves['fold_{}_class_{}'.format(f, cls)]['recall'] for f in range(3)]
        mean_recall = np.mean(class_recall, axis=0)
        class_score = [pr_curves['fold_{}_class_{}'.format(f, cls)]['average_precision'] for f in range(3)]
        mean_score = np.mean(class_score)

        mean_precisions[cls] = mean_precision
        mean_recalls[cls] = mean_recall
        mean_scores[cls] = mean_score

    lw = 2
    plt.figure(figsize=(6, 6))
    # plt.rcParams.update({'font.size': 14})

    cls = {0: 'NSCHL', 1: 'MCCHL', 2: 'LRCHL', 3: 'LDCHL'}
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'magenta'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(mean_recalls[i], mean_precisions[i], color=color, lw=lw,
                 label='{}  {:.3f}'.format(cls[i], mean_scores[i]))

    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    # plt.title('Receiver Operating Characteristic', fontsize=16)
    plt.legend(loc="lower right", fontsize=16)

    # Customizing the plot to only show 0 and 1 on the axes
    plt.xticks([0, 1], fontsize=12)
    plt.yticks([0, 1], fontsize=12)
    plt.grid(which='major', linestyle='--', linewidth='0.5', color='gray')

    # Making the box square
    plt.gca().set_aspect('equal', adjustable='box')

    plt.savefig(os.path.join(save_dir, 'inference/Average_PRcurve.png'))


def get_precision_recall(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """
    calculates precision-recall curve data from y true and prediction scores
    includes precision, recall, f1_score, thresholds, average_precision
    at each level of y, micro and macro averaged
    Args:
        y_true: true y values
        y_score: y prediction scores
    Returns:
        dict with precision-recall curve data
    """
    n_classes = 4
    y_true = label_binarize(y_true, classes=[i for i in range(n_classes)])

    # Compute PR curve and average precision for each class
    pr_curves = {}
    for i in range(n_classes):
        precision, recall, thresholds = metrics.precision_recall_curve(y_true[:, i], y_score[:, i])
        precision = np.delete(precision, -1)
        recall = np.delete(recall, -1)
        pr_curves[i] = {"precision": precision, "recall": recall, "thresholds": thresholds,
                        "average_precision": metrics.average_precision_score(y_true[:, i], y_score[:, i])}

    # Compute micro-average PR curve and average precision
    precision, recall, thresholds = metrics.precision_recall_curve(y_true.ravel(), y_score.ravel())
    precision = np.delete(precision, -1)
    recall = np.delete(recall, -1)
    pr_curves["micro"] = {"precision": precision, "recall": recall, "thresholds": thresholds,
                          "average_precision": metrics.average_precision_score(y_true, y_score, average="micro")}

    # Compute macro-average PR curve and average precision
    # First aggregate all false positive rates
    all_recall = np.unique(np.concatenate([pr_curves[i]["recall"] for i in range(n_classes)]))
    # Then interpolate all PR curves at this points
    mean_precision = np.zeros_like(all_recall)
    for i in range(n_classes):
        # xp needs to be increasing, but recall is decreasing, hence reverse the arrays
        mean_precision += interp(all_recall, pr_curves[i]["recall"][::-1], pr_curves[i]["precision"][::-1])
    # Finally average it and compute AUC
    mean_precision /= n_classes
    # reverse the arrays back
    all_recall = all_recall[::-1]
    mean_precision = mean_precision[::-1]
    pr_curves["macro"] = {"precision": mean_precision, "recall": all_recall,
                          "average_precision": metrics.average_precision_score(y_true, y_score, average="macro")}

    # calculate f1 score
    epsilon = 1e-7
    for i in pr_curves:
        precision = pr_curves[i]["precision"]
        recall = pr_curves[i]["recall"]
        pr_curves[i]["f1_score"] = 2 * (precision * recall) / (precision + recall + epsilon)

    return pr_curves


def display_PRcurve(epoch, pr_curves, n_classes, save_dir):
    lw = 2
    plt.figure(figsize=(12, 8), dpi=777)
    plt.rcParams.update({'font.size': 14})
    r"""plt.plot(pr_curves["micro"]["recall"], pr_curves["micro"]["precision"],
             label='micro-average PR curve (area = {0:0.3f})'
                   ''.format(pr_curves["micro"]["average_precision"]),
             color='navy', linestyle=':', linewidth=3)"""

    cls = {0: 'NSCHL', 1: 'MCCHL', 2: 'LRCHL', 3: 'LDCHL'}
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'magenta'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(pr_curves[i]["recall"], pr_curves[i]["precision"], color=color, lw=lw,
                 label='{}  {:.3f}'.format(cls[i], pr_curves[i]["average_precision"]))

    # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # plt.xticks(fontsize=13)
    # plt.yticks(fontsize=13)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, 'inference/fold_{}_PRcurve.png'.format(epoch)))
    # plt.show()


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]
