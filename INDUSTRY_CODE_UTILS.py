import math
import copy 
from tqdm import tqdm
from functools import partial
from torch.multiprocessing import Pool

import torch
import torch.nn.functional as F

from torch.nn.modules.loss import _WeightedLoss
from torch.optim.lr_scheduler import _LRScheduler


class NLP_Dataset(torch.utils.data.Dataset):
    def __init__(self, df, vocab, max_seq_length, pad_token, unk_token, train=True):
        self.word2idx = {term:idx for idx,term in enumerate(vocab)}
        self.idx2word = {idx:word for word,idx in self.word2idx.items()}
        
        self.pad_token, self.unk_token = pad_token, unk_token
        self.input_ids = []
        self.sequence_lens = []
        self.digit_1 = []
        self.digit_2 = []
        self.digit_3 = []
        self.train = train
        
        for i in range(df.shape[0]):
            input_ids, sequence_len = self.convert_text_to_input_ids(df.iloc[i].token, pad_to_len=max_seq_length)
            
            self.input_ids.append(input_ids.reshape(-1))
            self.sequence_lens.append(sequence_len)
            if self.train == True:
                self.digit_1.append(df.iloc[i].digit_1_encoded)
                self.digit_2.append(df.iloc[i].digit_2_encoded)
                self.digit_3.append(df.iloc[i].digit_3_encoded)
    
    def convert_text_to_input_ids(self, text, pad_to_len):
        words = text[:pad_to_len]
        deficit = pad_to_len - len(words)
        words.extend([self.pad_token]*deficit)
        for i in range(len(words)):
            if words[i] not in self.word2idx:
                words[i] = self.word2idx[self.unk_token]
            else:
                words[i] = self.word2idx[words[i]]
        return torch.Tensor(words).long(), pad_to_len - deficit

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, sample_id):
        sample_dict = dict()

        if self.train == True: 
            sample_dict['input_ids'] = self.input_ids[sample_id].reshape(-1)
            sample_dict['sequence_len'] = torch.tensor(self.sequence_lens[sample_id]).long()
            sample_dict['digit_1'] = torch.tensor(self.digit_1[sample_id])
            sample_dict['digit_2'] = torch.tensor(self.digit_2[sample_id])
            sample_dict['digit_3'] = torch.tensor(self.digit_3[sample_id])
            
        else:
            sample_dict['input_ids'] = self.input_ids[sample_id].reshape(-1)
            sample_dict['sequence_len'] = torch.tensor(self.sequence_lens[sample_id]).long()
            
        return sample_dict


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            

class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \
                .fill_(smoothing /(n_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
            self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss


class FocalLossWithSmoothing(torch.nn.Module):
    def __init__(
            self,
            num_classes: int,
            gamma: int = 1,
            lb_smooth: float = 0.1,
            ignore_index: int = None):

        super(FocalLossWithSmoothing, self).__init__()
        self._num_classes = num_classes
        self._gamma = gamma
        self._lb_smooth = lb_smooth
        self._ignore_index = ignore_index
        self._log_softmax = torch.nn.LogSoftmax(dim=1)

        if self._num_classes <= 1:
            raise ValueError('The number of classes must be 2 or higher')
        if self._gamma < 0:
            raise ValueError('Gamma must be 0 or higher')

    def forward(self, logits, label):

        logits = logits.float()
        difficulty_level = self._estimate_difficulty_level(logits, label)

        with torch.no_grad():
            label = label.clone().detach()
            if self._ignore_index is not None:
                ignore = label.eq(self._ignore_index)
                label[ignore] = 0
            lb_pos, lb_neg = 1. - self._lb_smooth, self._lb_smooth / (self._num_classes - 1)
            lb_one_hot = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()
        logs = self._log_softmax(logits)
        loss = -torch.sum(difficulty_level * logs * lb_one_hot, dim=1)
        if self._ignore_index is not None:
            loss[ignore] = 0
        return loss.mean()

    def _estimate_difficulty_level(self, logits, label):

        one_hot_key = torch.nn.functional.one_hot(label, num_classes=self._num_classes)
        if len(one_hot_key.shape) == 4:
            one_hot_key = one_hot_key.permute(0, 3, 1, 2)
        if one_hot_key.device != logits.device:
            one_hot_key = one_hot_key.to(logits.device)
        pt = one_hot_key * F.softmax(logits)
        difficulty_level = torch.pow(1 - pt, self._gamma)
        return difficulty_level
    
    
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
                )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)

def label_mask(label_dictionary):

    encoding = []
    for codes in label_dictionary.values():
        encoding.extend(codes)
    encoding = sorted(list(set(encoding)))

    decoding = {}
    for k, v in enumerate(encoding):
        decoding[k] = v

    mask = {}
    for k, values in label_dictionary.items():
        mask[k] = [True] * len(encoding)
        for v in values:
            mask[k][encoding.index(v)] = False
            
    return mask 

def calc_accuracy(X,Y):
    _, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc


def calc_accuracy_ALL(X_1,X_2,X_3,Y_1,Y_2,Y_3):
    _, max_indices_1 = torch.max(X_1, 1)
    _, max_indices_2 = torch.max(X_2, 1)
    _, max_indices_3 = torch.max(X_3, 1)
    running_sum = (max_indices_1 == Y_1).sum().data.cpu().numpy() + (max_indices_2 == Y_2).sum().data.cpu().numpy() + (max_indices_3 == Y_3).sum().data.cpu().numpy()
    running_size = max_indices_1.size()[0] + max_indices_2.size()[0] + max_indices_3.size()[0]
    train_acc = running_sum / running_size
    return train_acc


def masking(x, digit_mask):
    return digit_mask[x]


def predict(models, dataset, config):
    results_1 = []
    results_2 = []
    results_3 = []
    tqdm_dataset = tqdm(enumerate(dataset), total=len(dataset))
    for _, batch_item in tqdm_dataset:
        for fold,model in enumerate(models):
            model.eval()
            with torch.no_grad():
                if fold == 0:
                    pred_1, pred_2, pred_3 = model(batch_item)
                else:
                    pred_1_, pred_2_, pred_3_ = model(batch_item)
                    
                    pred_1 = pred_1 + pred_1_
                    pred_2 = pred_2 + pred_2_
                    pred_3 = pred_3 + pred_3_
        pred_1 = 0.2*pred_1
        pred_2 = 0.2*pred_2
        pred_3 = 0.2*pred_3
        
        _, max_indices_1 = torch.max(pred_1, 1)
        with Pool(1) as pool:
            mask_1 = pool.map(partial(masking, digit_mask=config['digit_1_to_2_mask']), max_indices_1.tolist())
        mask_1 = torch.Tensor(mask_1).bool().to(config['model_device'])
        pred_2 = pred_2.masked_fill_(mask_1, -10000.)

        _, max_indices_2 = torch.max(pred_2, 1)
        with Pool(1) as pool:
            mask_2 = pool.map(partial(masking, digit_mask=config['digit_2_to_3_mask']), max_indices_2.tolist())
        mask_2 = torch.Tensor(mask_2).bool().to(config['model_device'])
        pred_3 = pred_3.masked_fill_(mask_2, -10000.)
        
        pred_1 = torch.tensor(torch.argmax(pred_1, axis=-1), dtype=torch.int32).cpu().numpy()
        pred_2 = torch.tensor(torch.argmax(pred_2, axis=-1), dtype=torch.int32).cpu().numpy()
        pred_3 = torch.tensor(torch.argmax(pred_3, axis=-1), dtype=torch.int32).cpu().numpy()
        results_1.extend(pred_1)
        results_2.extend(pred_2)
        results_3.extend(pred_3)
        
    return results_1, results_2, results_3