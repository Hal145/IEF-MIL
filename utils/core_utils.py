import numpy as np
import torch
import os
from utils.utils import *
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB, MultiScaleCLAM_MB
import models.dsmil as DSMIL
from models.transmil import TransMIL
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from datasets.dataset_generic import save_splits

from functools import partial

from models.transformer import ACMIL_GA
from models.transformer import ACMIL_MHA

from loss_functions.focal_loss import SparseCategoricalFocalLoss as focal_loss

class Config:
    def __init__(self, D_feat, D_inner, n_class, n_token, n_masked_patch, mask_drop):
        self.D_feat = D_feat
        self.D_inner = D_inner
        self.n_class = n_class
        self.n_token = n_token
        self.n_masked_patch = n_masked_patch
        self.mask_drop = mask_drop

class Accuracy_Logger(object):
    """Accuracy logger"""

    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count

        return acc, correct, count


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=15, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


def train(no_validation, datasets, cur, args):
    device = args.device
    """
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    if no_validation==False:
        print('\nInit train/test splits...', end=' ')
        train_split, test_split = datasets
        save_splits(datasets, ['train', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
        print('Done!')
        print("Training on {} samples".format(len(train_split)))
        print("Testing on {} samples".format(len(test_split)))
    else:
        print('\nInit train/val/test splits...', end=' ')
        train_split, val_split, test_split = datasets
        save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
        print('Done!')
        print("Training on {} samples".format(len(train_split)))
        print("Validating on {} samples".format(len(val_split)))
        print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes=args.n_classes)
        if device == device:
            loss_fn = loss_fn.cuda()
    elif args.bag_loss == 'focal':
        loss_fn = focal_loss(gamma=2, from_logits=True)
        loss_fn = loss_fn.to(device)
    else:
        bag_loss_weights = list(map(float, args.bag_loss_weights.split(',')))
        weights=torch.tensor(bag_loss_weights)
        loss_fn = nn.CrossEntropyLoss()  # weight=weights, label_smoothing=0.1
        loss_fn = loss_fn.to(device)
    print('Done!')

    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}

    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})

    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})

        if args.B > 0:
            model_dict.update({'k_sample': args.B})

        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes=2)
            if device == device:
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()

        if args.model_type == 'clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn).to(device)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn).to(device)
        elif args.model_type == 'clam_mb_ms':
            model = MultiScaleCLAM_MB(**model_dict).to(device)
        else:
            raise NotImplementedError

    elif args.model_type == 'dsmil':
        i_classifier = DSMIL.FCLayer(in_size=512, out_size=4).to(device)
        b_classifier = DSMIL.BClassifier(input_size=512, output_class=4, dropout_v=0, nonlinear=True).to(device)
        model = DSMIL.MILNet(i_classifier, b_classifier).to(device)

    elif args.model_type == 'transmil':
        model = TransMIL(n_classes=4, dim=768).to(device)

    elif args.model_type == 'acmil_ga':
        conf = Config(D_feat=1024, D_inner=512, n_class=4, n_token=5, n_masked_patch=10, mask_drop=0.6)
        model = ACMIL_GA(conf, n_token=conf.n_token, n_masked_patch=conf.n_token, mask_drop=conf.mask_drop)
        model = model.to(device)
    elif args.model_type == 'acmil_mha':
        conf = Config(D_feat=512, D_inner=256, n_class=4, n_token=5, n_masked_patch=10, mask_drop=0.6)
        model = ACMIL_MHA(conf, n_token=conf.n_token, n_masked_patch=conf.n_token, mask_drop=conf.mask_drop)
        model = model.to(device)

    else:  # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)


    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    if no_validation==False:
        print('\nInit Loaders...', end=' ')
        train_loader = get_split_loader(train_split, training=True, testing=args.testing, weighted=args.weighted_sample)
        test_loader = get_split_loader(test_split, testing=args.testing)
        print('Done!')
    else:
        print('\nInit Loaders...', end=' ')
        train_loader = get_split_loader(train_split, training=True, testing=args.testing, weighted=args.weighted_sample)
        val_loader = get_split_loader(val_split, testing=args.testing)
        test_loader = get_split_loader(test_split, testing=args.testing)
        print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping and no_validation==True:
        early_stopping = EarlyStopping(patience=15, stop_epoch=50, verbose=True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_mb']:
            if not args.no_inst_cluster and args.no_validation==True:
                train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, args.model_type,
                                writer, loss_fn, args.device)
                stop = validate_clam(cur, epoch, model, val_loader, args.n_classes, args.model_type,
                                     early_stopping, writer, loss_fn, args.results_dir, args.device)
            elif not args.no_inst_cluster and args.no_validation==False:
                train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, args.model_type,
                                writer, loss_fn, args.device)
        else:
            if args.no_validation==False:
                train_loop(epoch, model, train_loader, optimizer, args.n_classes, args.model_type, writer, loss_fn, args.device)
            else:
                train_loop(epoch, model, train_loader, optimizer, args.n_classes, args.model_type, writer, loss_fn, args.device)
                stop = validate(cur, epoch, model, val_loader, args.n_classes, args.model_type,
                                early_stopping, writer, loss_fn, args.results_dir, args.device)

                if stop:
                    break

        if epoch%20==0:
            torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_{}_checkpoint.pt".format(cur, epoch)))

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    if no_validation==True:
        _, val_error, val_auc, _ = summary(model, val_loader, args.n_classes, args.model_type, args.device)
        print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes, args.model_type, args.device)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        if no_validation==True:
            writer.add_scalar('final/val_error', val_error, 0)
            writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()

    if no_validation==False:
        return results_dict, test_auc, 1 - test_error,
    else:
        return results_dict, test_auc, val_auc, 1 - test_error, 1 - val_error



def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, model_type, writer=None, loss_fn=None,
                    device_=None):
    model.train()
    device = device_

    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)

    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    for batch_idx, (data_125x, data_25x, data_5x, data_10x, data_20x, label, slide_id) in enumerate(loader):
        data_5x = data_5x.to(device)
        data_10x = data_10x.to(device)
        data_20x = data_20x.to(device)
        label = label.to(device)
        data = torch.cat((data_5x, data_10x, data_20x), dim=0)
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

        acc_logger.log(Y_hat, label)
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count += 1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value

        total_loss = bag_weight * loss + (1 - bag_weight) * instance_loss

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value,
                                                                                                  instance_loss_value,
                                                                                                  total_loss.item()) +
                  'label: {}, bag_size: {}'.format(label.item(), data_5x.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error

        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(4):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss,
                                                                                                      train_inst_loss,
                                                                                                      train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)


def train_loop(epoch, model, loader, optimizer, n_classes, model_type, writer=None, loss_fn=None, device_=None):
    device = device_

    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data_125x, data_25x, data_5x, data_10x, data_20x, label, slide_id) in enumerate(loader):
        data_5x = data_5x.to(device)
        data_10x = data_10x.to(device)
        data_20x = data_20x.to(device)

        if model_type=='clam_mb_ms':
            label = label.to(device)
            logits, Y_prob, Y_hat, _, instance_dict = model((data_125x, data_25x, data_5x, data_10x, data_20x), label=label,
                                                            instance_eval=True)

        elif model_type == 'transmil':
            data = torch.cat((data_5x, data_10x, data_20x), dim=0)
            label = label.to(device)
            logits, Y_prob, Y_hat = model(data)

        elif model_type == 'dsmil':
            data = torch.cat((data_10x, data_20x), dim=0)
            label = label.to(device)
            ins_prediction, bag_prediction, _, _ = model(data_10x)
            # max_prediction, _ = torch.max(ins_prediction, 0)
            # top_10 = torch.topk(ins_prediction, 10, dim=0, largest=True)[0]
            max_prediction = torch.mean(ins_prediction, dim=0)  # top_10
            bag_loss = loss_fn(bag_prediction, label)
            max_loss = loss_fn(max_prediction.unsqueeze(0), label)
            loss_dsmil = 0.5 * bag_loss + 0.5 * max_loss
            Y_prob = (0.5 * torch.softmax(max_prediction.unsqueeze(0), dim=1) + 0.5 * torch.softmax(bag_prediction, dim=1))
            Y_hat = torch.argmax(Y_prob, dim=1)

        elif model_type == 'acmil_ga' or model_type == 'acmil_mha':
            data = torch.cat((data_5x, data_10x, data_20x), dim=0)
            label = label.to(device)
            sub_preds, slide_preds, attn = model(data)
            Y_prob = torch.softmax(slide_preds, dim=-1)
            Y_hat = torch.argmax(Y_prob, dim=1)

        else:
            label = label.to(device)
            logits, Y_prob, Y_hat, _, instance_dict = model([data_125x, data_25x, data_5x, data_10x, data_20x])


        acc_logger.log(Y_hat, label)
        if model_type=='dsmil':
            loss = loss_dsmil
            # print(Y_prob.device)
            # loss = loss_fn(Y_prob, label)

        elif model_type=='acmil_ga' or model_type=='acmil_mha':
            if model_type=='acmil_ga':
                attn = attn.squeeze(dim=0)
            else:
                attn = attn
            n_token = 5
            loss0 = loss_fn(sub_preds, label.repeat_interleave(n_token))
            loss1 = loss_fn(slide_preds, label)

            diff_loss = torch.tensor(0).to(device, dtype=torch.float)
            attn = torch.softmax(attn, dim=-1)

            for i in range(n_token):
                for j in range(i + n_token):
                    diff_loss += torch.cosine_similarity(attn[:, i], attn[:, j], dim=-1).mean() / (
                            n_token * (n_token - 1) / 2)

            loss = diff_loss + loss0 + loss1

        else:
            loss = loss_fn(logits, label)
        loss_value = loss.item()

        # print('Scale scores', instance_dict)
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_sizes: {},{},{},{}'.format(batch_idx, loss_value,
                                                                                           label.item(),
                                                                           data_5x.size(0), data_10x.size(0),
                                                                           data_20x.size(0), slide_id))

        # print('Results by scale', instance_dict)

        error = calculate_error(Y_hat, label)
        train_error += error

        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)


def validate(cur, epoch, model, loader, n_classes, model_type, early_stopping=None, writer=None, loss_fn=None,
             results_dir=None, device_=None):
    device = device_
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    val_loss_values={}
    with torch.no_grad():
        for batch_idx, (data_125x, data_25x, data_5x, data_10x, data_20x, label, slide_id) in enumerate(loader):

            data_5x = data_5x.to(device, non_blocking=True)
            data_10x = data_10x.to(device, non_blocking=True)
            data_20x = data_20x.to(device, non_blocking=True)

            if model_type == 'clam_mb_ms':
                label = label.to(device, non_blocking=True)
                logits, Y_prob, Y_hat, _, instance_dict = model((data_125x, data_25x, data_5x, data_10x, data_20x), label=label,
                                                                instance_eval=True)
            elif model_type == 'dsmil':
                data = torch.cat((data_10x, data_20x), dim=0)
                label = label.to(device, non_blocking=True)
                ins_prediction, bag_prediction, _, _ = model(data_10x)
                # max_prediction, _ = torch.max(ins_prediction, 0)
                # top_10 = torch.topk(ins_prediction, 10, dim=0, largest=True)[0]
                max_prediction = torch.mean(ins_prediction, dim=0)  # top_10
                bag_loss = loss_fn(bag_prediction, label)
                max_loss = loss_fn(max_prediction.unsqueeze(0), label)
                loss_dsmil = 0.5 * bag_loss + 0.5 * max_loss
                Y_prob = (0.5 * torch.softmax(max_prediction.unsqueeze(0), dim=1) + 0.5 * torch.softmax(bag_prediction,
                                                                                                        dim=1))
                Y_hat = torch.argmax(Y_prob, dim=1)

            elif model_type == 'transmil':
                data = torch.cat((data_5x, data_10x, data_20x), dim=0)
                label = label.to(device, non_blocking=True)
                logits, Y_prob, Y_hat = model(data)

            elif model_type == 'acmil_ga' or model_type == 'acmil_mha':
                data = torch.cat((data_5x, data_10x, data_20x), dim=0)
                label = label.to(device)
                sub_preds, slide_preds, attn = model(data)
                Y_prob = torch.softmax(slide_preds, dim=-1)
                Y_hat = torch.argmax(Y_prob, dim=1)

            else:
                label = label.to(device, non_blocking=True)
                logits, Y_hat, Y_prob, _, instance_dict = model([data_125x, data_25x, data_5x, data_10x, data_20x])


            acc_logger.log(Y_hat, label)
            if model_type == 'dsmil':
                loss = loss_dsmil
                # loss = loss_fn(Y_prob, label)

            elif model_type == 'acmil_ga' or model_type == 'acmil_mha':
                loss = loss_fn(slide_preds, label)

            else:
                loss = loss_fn(logits, label)

            val_loss_values[slide_id]=loss

            # print('Results by scale', instance_dict)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])

    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    # print('Validation set one by one loss ------------------------------------> {}'.format(val_loss_values))
    # print('-------------------------------------------------------------------')
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def validate_clam(cur, epoch, model, loader, n_classes, model_type, early_stopping=None, writer=None, loss_fn=None,
                  results_dir=None, device_=None):
    device = device_
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count = 0

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.no_grad():
        for batch_idx, (data_125x, data_25x, data_5x, data_10x, data_20x, label, slide_id) in enumerate(loader):
            data_20x, data_10x, data_5x, label = data_20x.to(device), data_10x.to(device), data_5x.to(device), label.to(device)
            data = torch.cat((data_5x, data_10x, data_20x), dim=0)
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)

            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']

            inst_count += 1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary(model, loader, n_classes, model_type, device_=None):
    device = device_
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data_125x, data_25x, data_5x, data_10x, data_20x, label, slide_id) in enumerate(loader):
        # data_125x = data_125x.to(device)
        # data_25x = data_25x.to(device)
        data_5x = data_5x.to(device)
        data_10x = data_10x.to(device)
        data_20x = data_20x.to(device)
        label = label.to(device)

        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():

            if model_type == 'clam_mb_ms':
                label = label.to(device)
                logits, Y_prob, Y_hat, _, instance_dict = model((data_125x, data_25x, data_5x, data_10x, data_20x), label=label,
                                                                instance_eval=True)
            elif model_type == 'dsmil':
                data = torch.cat((data_10x, data_20x), dim=0)
                label = label.to(device)
                ins_prediction, bag_prediction, _, _ = model(data_20x)
                # max_prediction, _ = torch.max(ins_prediction, 0)
                # top_10 = torch.topk(ins_prediction, 10, dim=0, largest=True)[0]
                max_prediction = torch.mean(ins_prediction, dim=0)  # top_10
                Y_prob = (0.5 * torch.softmax(max_prediction.unsqueeze(0), dim=1) + 0.5 * torch.softmax(bag_prediction, dim=1))
                Y_hat = torch.argmax(Y_prob, dim=1)

            elif model_type == 'transmil':
                data = torch.cat((data_5x, data_10x, data_20x), dim=0)
                label = label.to(device)
                logits, Y_prob, Y_hat = model(data)

            elif model_type == 'acmil_ga' or model_type == 'acmil_mha':
                data = torch.cat((data_5x, data_10x, data_20x), dim=0)
                label = label.to(device)
                sub_preds, slide_preds, attn = model(data)
                Y_prob = torch.softmax(slide_preds, dim=-1)
                Y_hat = torch.argmax(Y_prob, dim=1)

            elif model_type == 'clam_mb' or model_type == 'clam_sb':
                data = torch.cat((data_5x, data_10x, data_20x), dim=0)
                logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

            else:
                label = label.to("cuda:1")
                logits, Y_prob, Y_hat, _, instance_dict = model([data_125x, data_25x, data_5x, data_10x, data_20x])

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

        # print('Results by scale', instance_dict)

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    return patient_results, test_error, auc, acc_logger
