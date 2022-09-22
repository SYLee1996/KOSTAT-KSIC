import os 
import time
import copy 
import click
import random
import datetime
import collections
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from INDUSTRY_CODE_MODEL import LSTMEncoder
from INDUSTRY_CODE_UTILS import NLP_Dataset, SmoothCrossEntropyLoss, FocalLossWithSmoothing, EarlyStopping, CosineAnnealingWarmUpRestarts, calc_accuracy_ALL, label_mask

import warnings
warnings.filterwarnings(action='ignore')

import wandb

@click.command()
@click.option('--model', type=click.STRING, default='LSTM', required=True)
@click.option('--batch_size', type=click.STRING, default=512, required=True)
@click.option('--max_seq_len', type=click.STRING, default=128, required=True)
@click.option('--dropout', type=click.STRING, default=0.3, required=True)
@click.option('--hidden_size', type=click.STRING, default=2048, required=False)
@click.option('--hidden_size_2', type=click.STRING, default=512, required=False)
@click.option('--lstm_unit_cnt', type=click.STRING, default=2, required=True)
@click.option('--label_smoothing', type=click.STRING, default=0.3, required=False)
@click.option('--freeze_embeddings', type=click.STRING, default=False, required=True)
@click.option('--patience', type=click.STRING, default=5, required=True)
@click.option('--n_fold', type=click.STRING, default=5, required=True)
@click.option('--optimizer', type=click.STRING, default='AdamW', required=True)
@click.option('--loss_function', type=click.STRING, default='CE_with_Lb', required=True)
@click.option('--device', type=click.STRING, default='0,1,2,3', required=True)
@click.option('--weight_decay', type=click.STRING, default=0.0001, required=True)
@click.option('--epochs', type=click.STRING, default=100, required=True)
@click.option('--train_load', type=click.STRING, required=True)
@click.option('--vocab_load', type=click.STRING, required=True)
@click.option('--embedd_load', type=click.STRING, required=True)
@click.option('--save_name', type=click.STRING, required=True)
@click.option('--weighted_sum', type=click.STRING, default=True, required=True)
@click.option('--sampling_num', type=click.STRING, default='ALL', required=True)

@click.option('--lr_scheduler', type=click.STRING, default='CosineAnnealingLR', required=True)
@click.option('--lr', type=click.STRING, default=1e-3, required=True)
@click.option('--lr_t', type=click.STRING, default=3, required=True)
@click.option('--gamma', type=click.STRING, default=0.5, required=True)

def main(model, batch_size, max_seq_len, dropout, hidden_size, hidden_size_2, lstm_unit_cnt, label_smoothing,
        train_load, vocab_load, embedd_load, freeze_embeddings, patience, n_fold, optimizer, loss_function, 
        weighted_sum, sampling_num, weight_decay, epochs, device, save_name,
        lr, lr_t, gamma, lr_scheduler):
    
    seed = 10
    vocab_npa = np.load(vocab_load)
    embs_npa = np.load(embedd_load)
    try:
        eval(sampling_num)
        sampling_num = int(sampling_num)
    except:
        sampling_num = sampling_num
        
    suffix = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime("%y%m%d_%H%M")
    
    config = {
        #model configurations
        'model': model, 
        'batch_size': int(batch_size),
        'max_seq_len': int(max_seq_len),
        'lr': float(lr),
        'dropout': float(dropout),
        'hidden_size': int(hidden_size),
        'hidden_size_2': int(hidden_size_2),
        'lstm_unit_cnt': int(lstm_unit_cnt),
        'label_smoothing': float(label_smoothing),
        'optimizer': optimizer,
        'loss_function': loss_function,
        'weight_decay': float(weight_decay),
        'patience': int(patience),
        'device': device,
        'lr_t': int(lr_t),
        'gamma': float(gamma),
        'lr_scheduler': lr_scheduler,
        
        #embeddings configurations
        'pretrained_embeddings': embs_npa,
        'freeze_embeddings': eval(freeze_embeddings),
        'vocab': vocab_npa,
        'PAD_TOK':'<pad>',
        'UNK_TOK':'<unk>',

        #training
        'epochs': int(epochs),
        'n_fold': int(n_fold),
        'weighted_sum': eval(weighted_sum),
        'sampling_num': sampling_num
        }
    
    model_save_name ='./RESULTS/'+save_name+"_"+suffix+"(" +str(config['batch_size'])+"_"+\
                                                            str(config['max_seq_len'])+"_"+\
                                                            str(config['lr'])+"_"+\
                                                            str(config['dropout'])+"_"+\
                                                            str(config['hidden_size'])+"_"+\
                                                            str(config['hidden_size_2'])+"_"+\
                                                            str(config['lstm_unit_cnt'])+"_"+\
                                                            str(config['label_smoothing'])+"_"+\
                                                            str(config['optimizer'])+"_"+\
                                                            str(config['loss_function'])+"_"+\
                                                            str(config['weight_decay'])+"_"+\
                                                            str(config['freeze_embeddings'])+"_"+\
                                                            str(config['lr_scheduler'])+"_"+\
                                                            str(config['lr_t'])+"_"+\
                                                            str(config['gamma'])+"_"+\
                                                            str(config['weighted_sum'])+"_"+\
                                                            str(config['sampling_num'])+")"                                     
                
    config['model_save_name'] = model_save_name
    # -------------------------------------------------------------------------------------------
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device']
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    print('Device: %s' % device)
    if (device.type == 'cuda') or (torch.cuda.device_count() > 1):
        print('GPU activate --> Count of using GPUs: %s' % torch.cuda.device_count())
    config['model_device'] = device
        
    # 데이터 로드-----------------------------------------------------------------------------------

    label_data = pd.read_excel('한국표준산업분류(10차)_국문.xlsx',header=[0,1,2])
    label_data = label_data.fillna(method='ffill')
    label_data = label_data['개정 분류체계(제10차 기준)'][['대분류(21)', '중분류(77)', '소분류(232)']]

    digit_1 = list(label_data['대분류(21)']['코드'].dropna().unique())
    digit_2 = list(label_data['중분류(77)']['코드'].dropna().unique())
    digit_2 = list(map(int, digit_2))
    digit_3 = list(label_data['소분류(232)']['코드'].dropna().unique())
    digit_3 = list(map(int, digit_3))

    # label encoding

    digit_1_encoder = LabelEncoder()
    digit_1_encoder.fit(digit_1)

    digit_2_encoder = LabelEncoder()
    digit_2_encoder.fit(digit_2)

    digit_3_encoder = LabelEncoder()
    digit_3_encoder.fit(digit_3)
    
    config['digit_1_num'] = len(digit_1)
    config['digit_2_num'] = len(digit_2)
    config['digit_3_num'] = len(digit_3)

    # ------------------------------------------------------------------------------------------------
    
    code_df = pd.concat([label_data['대분류(21)']['코드'], 
                        label_data['중분류(77)']['코드'], 
                        label_data['소분류(232)']['코드']],axis=1)
    code_df.columns = ['digit_1', 'digit_2', 'digit_3']

    code_df['digit_2'] = code_df['digit_2'].map(int)
    code_df['digit_3'] = code_df['digit_3'].map(int)

    code_df['digit_1'] = digit_1_encoder.transform(code_df['digit_1'])
    code_df['digit_2'] = digit_2_encoder.transform(code_df['digit_2'])
    code_df['digit_3'] = digit_3_encoder.transform(code_df['digit_3'])

    digit_1_to_digit_2 = [dict(zip([key],[value])) for key, value in zip(code_df['digit_1'], code_df['digit_2'])]
    digit_2_to_digit_3 = [dict(zip([key],[value])) for key, value in zip(code_df['digit_2'], code_df['digit_3'])]

    digit_1_to_digit_2_list = list(map(dict, collections.OrderedDict.fromkeys(tuple(sorted(d.items())) for d in digit_1_to_digit_2)))
    digit_2_to_digit_3_list = list(map(dict, collections.OrderedDict.fromkeys(tuple(sorted(d.items())) for d in digit_2_to_digit_3)))

    digit_1_to_digit_2_dict = collections.defaultdict(list)
    digit_2_to_digit_3_dict = collections.defaultdict(list)

    for digit_list, digit_dict in zip((digit_1_to_digit_2_list, digit_2_to_digit_3_list), (digit_1_to_digit_2_dict, digit_2_to_digit_3_dict)):
        for d2 in digit_list:
            for k, v in d2.items():
                digit_dict[k].append(v)
                
    config['digit_1_to_2_mask'] = label_mask(digit_1_to_digit_2_dict)
    config['digit_2_to_3_mask'] = label_mask(digit_2_to_digit_3_dict)
    
    # ------------------------------------------------------------------------------------------------

    train_data_tokenized_npy = np.load(train_load, allow_pickle=True)
    train = pd.DataFrame(train_data_tokenized_npy, columns = ['digit_1_encoded', 'digit_2_encoded', 'digit_3_encoded', 
                                                            'token'])
    
    # train -------------------------------------------------------------------------------------
    train['digit_1_encoded'] = train['digit_1_encoded'].astype(int)
    train['digit_2_encoded'] = train['digit_2_encoded'].astype(int)
    train['digit_3_encoded'] = train['digit_3_encoded'].astype(int)
    
    if config['sampling_num'] == 'ALL':
        sampled_train = train
        remained_df = pd.DataFrame()
    else:
        sampled_train = train.sample(frac=1.0)
        sampled_train = sampled_train.groupby("digit_3_encoded").head(config['sampling_num'])
        a_sub_b = [x for x in train.index if x not in sampled_train.index]
        remained_df = train.iloc[a_sub_b]
        
    # Cross Validation
    kfold = StratifiedKFold(n_splits=config['n_fold'],shuffle=True,random_state=seed)
    n_fold = config['n_fold']
    k_valid_acc_plot, k_valid_f1_plot = [], []   
    wandb_name = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime("%y%m%d_%H%M%S")
    
    print(config['model_save_name']+ ' is start!')
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(sampled_train, sampled_train['digit_3_encoded'])):

        Train_set = sampled_train.iloc[train_idx]
        Valid_set = sampled_train.iloc[valid_idx]
        Valid_set = pd.concat([Valid_set, remained_df])

        wandb.init(project='KOSTAT_BEST', name=wandb_name, entity="sylee1996", config=config, reinit=True)
        wandb_config = wandb.config

        # Train
        train_dataset = NLP_Dataset(Train_set, 
                            config['vocab'],
                            wandb_config['max_seq_len'], wandb_config['PAD_TOK'],
                            wandb_config['UNK_TOK'], train=True)
        Train_loader = DataLoader(train_dataset, batch_size=wandb_config['batch_size'], 
                                num_workers=16, prefetch_factor=config['batch_size']*2,
                                shuffle=True, drop_last=False, pin_memory=True)
        
        # Validation 
        valid_dataset = NLP_Dataset(Valid_set, 
                            config['vocab'],
                            wandb_config['max_seq_len'], wandb_config['PAD_TOK'],
                            wandb_config['UNK_TOK'], train=True)
        Valid_loader = DataLoader(valid_dataset, batch_size=wandb_config['batch_size'],
                                num_workers=16, prefetch_factor=config['batch_size']*2,
                                shuffle=True, drop_last=False, pin_memory=True)

        model = LSTMEncoder(wandb_config, config).to(device)   
        model = nn.DataParallel(model).to(device)

        if wandb_config['lr_scheduler'] == 'CosineAnnealingLR':
            optimizer = torch.optim.AdamW(model.parameters(), lr=wandb_config['lr'], weight_decay=wandb_config['weight_decay'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=wandb_config['lr_t'], eta_min=0)
            
        elif wandb_config['lr_scheduler'] == 'CosineAnnealingWarmUpRestarts':
            optimizer = torch.optim.AdamW(model.parameters(), lr=0, weight_decay=wandb_config['weight_decay'])
            scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=wandb_config['lr_t'], eta_max=wandb_config['lr'], gamma=wandb_config['gamma'], T_mult=1, T_up=0)
        
        scaler = torch.cuda.amp.GradScaler()

        if wandb_config['loss_function'] == 'Focal_with_Lb':
            loss_fn_digit_1 = FocalLossWithSmoothing(num_classes=wandb_config['digit_1_num'],
                                                    lb_smooth=wandb_config['label_smoothing'],
                                                    gamma=2)
            
            loss_fn_digit_2 = FocalLossWithSmoothing(num_classes=wandb_config['digit_2_num'],
                                                    lb_smooth=wandb_config['label_smoothing'],
                                                    gamma=2)
            
            loss_fn_digit_3 = FocalLossWithSmoothing(num_classes=wandb_config['digit_3_num'],
                                                    lb_smooth=wandb_config['label_smoothing'],
                                                    gamma=2)
            
        elif wandb_config['loss_function'] == 'CE_with_Lb':
            loss_fn = SmoothCrossEntropyLoss(smoothing=wandb_config['label_smoothing'])

        early_stopping = EarlyStopping(patience=wandb_config['patience'], mode='max')

        wandb.watch(model)
        best=0.5
        valid_acc_plot, valid_f1_plot = [], []
        epochs = wandb_config['epochs']

        for e in range(epochs):
            start=time.time()
            train_loss, valid_loss = 0.0, 0.0  
            train_accuracy, valid_accuracy = 0.0, 0.0
            train_acc_digit_1, train_acc_digit_2, train_acc_digit_3 = 0.0, 0.0, 0.0
            valid_acc_digit_1, valid_acc_digit_2, valid_acc_digit_3 = 0.0, 0.0, 0.0
                    
            model.train()
            for batch_id, batch in tqdm(enumerate(Train_loader), total=len(Train_loader)):
                optimizer.zero_grad()
                train_x = batch
                train_y_1 = batch['digit_1'].to(device)
                train_y_2 = batch['digit_2'].to(device)
                train_y_3 = batch['digit_3'].to(device)
                        
                with torch.cuda.amp.autocast():    
                    pred_1, pred_2, pred_3 = model(train_x)
                    pred_1 = pred_1.type(torch.FloatTensor).to(device)
                    pred_2 = pred_2.type(torch.FloatTensor).to(device)
                    pred_3 = pred_3.type(torch.FloatTensor).to(device)
                    
                    if wandb_config['loss_function'] == 'Focal_with_Lb':
                        loss_1 = loss_fn_digit_1(pred_1, train_y_1)
                        loss_2 = loss_fn_digit_2(pred_2, train_y_2)
                        loss_3 = loss_fn_digit_3(pred_3, train_y_3)
                        
                    elif wandb_config['loss_function'] == 'CE_with_Lb':
                        loss_1 = loss_fn(pred_1, train_y_1)
                        loss_2 = loss_fn(pred_2, train_y_2)
                        loss_3 = loss_fn(pred_3, train_y_3)
                        
                    if wandb_config['weighted_sum']  == True:
                        loss = 0.2*loss_1 + 0.3*loss_2 + 0.5*loss_3
                    else:
                        loss = loss_1 + loss_2 + loss_3
                        
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    train_loss += loss.detach().cpu().numpy()
                    train_acc_digit_1 += f1_score(train_y_1.detach().cpu().numpy(), torch.max(pred_1, 1)[1].cpu().numpy(), average='macro')
                    train_acc_digit_2 += f1_score(train_y_2.detach().cpu().numpy(), torch.max(pred_2, 1)[1].cpu().numpy(), average='macro')
                    train_acc_digit_3 += f1_score(train_y_3.detach().cpu().numpy(), torch.max(pred_3, 1)[1].cpu().numpy(), average='macro')
                    train_accuracy += calc_accuracy_ALL(pred_1, pred_2, pred_3, train_y_1, train_y_2, train_y_3)

            train_loss = train_loss/(batch_id+1)
            train_acc_digit_1 = train_acc_digit_1/(batch_id+1)
            train_acc_digit_2 = train_acc_digit_2/(batch_id+1)
            train_acc_digit_3 = train_acc_digit_3/(batch_id+1)
            train_accuracy = train_accuracy/(batch_id+1)
            train_F1 = np.mean([train_acc_digit_1,train_acc_digit_2,train_acc_digit_3])
            
            scheduler.step()
            
            model.eval()
            for valid_batch_id, batch in tqdm(enumerate(Valid_loader), total=len(Valid_loader)):
                with torch.no_grad():
                    valid_x = batch
                    valid_y_1 = batch['digit_1'].to(device)
                    valid_y_2 = batch['digit_2'].to(device)
                    valid_y_3 = batch['digit_3'].to(device)
                    
                    pred_1, pred_2, pred_3 = model(valid_x, valid=False)
                    pred_1 = pred_1.type(torch.FloatTensor).to(device)
                    pred_2 = pred_2.type(torch.FloatTensor).to(device)
                    pred_3 = pred_3.type(torch.FloatTensor).to(device)
                    
                    if wandb_config['loss_function'] == 'Focal_with_Lb':
                        val_loss_1 = loss_fn_digit_1(pred_1, valid_y_1)
                        val_loss_2 = loss_fn_digit_2(pred_2, valid_y_2)
                        val_loss_3 = loss_fn_digit_3(pred_3, valid_y_3)
                        
                    elif wandb_config['loss_function'] == 'CE_with_Lb':
                        val_loss_1 = loss_fn(pred_1, valid_y_1)
                        val_loss_2 = loss_fn(pred_2, valid_y_2)
                        val_loss_3 = loss_fn(pred_3, valid_y_3)
                        
                if wandb_config['weighted_sum']  == True:
                    val_loss = 0.2*val_loss_1 + 0.3*val_loss_2 + 0.5*val_loss_3 
                else:
                    val_loss = val_loss_1 + val_loss_2 + val_loss_3
                        
                valid_loss += val_loss.detach().cpu().numpy()
                valid_acc_digit_1 += f1_score(valid_y_1.detach().cpu().numpy(), torch.max(pred_1, 1)[1].cpu().numpy(), average='macro')
                valid_acc_digit_2 += f1_score(valid_y_2.detach().cpu().numpy(), torch.max(pred_2, 1)[1].cpu().numpy(), average='macro')
                valid_acc_digit_3 += f1_score(valid_y_3.detach().cpu().numpy(), torch.max(pred_3, 1)[1].cpu().numpy(), average='macro')
                valid_accuracy += calc_accuracy_ALL(pred_1, pred_2, pred_3, valid_y_1, valid_y_2, valid_y_3)


            valid_loss = valid_loss/(valid_batch_id+1)
            valid_acc_digit_1 = valid_acc_digit_1/(valid_batch_id+1)
            valid_acc_digit_2 = valid_acc_digit_2/(valid_batch_id+1)
            valid_acc_digit_3 = valid_acc_digit_3/(valid_batch_id+1)
            valid_accuracy = valid_accuracy/(valid_batch_id+1)
            valid_F1 = np.mean([valid_acc_digit_1,valid_acc_digit_2,valid_acc_digit_3])
            
            valid_acc_plot.append(valid_accuracy)
            valid_f1_plot.append(valid_F1)
            
            wandb.log({
                    "Epoch": e+1,
                    "train_loss": train_loss,
                    'train_accuracy': train_accuracy,
                    "train_F1": train_F1,
                    
                    "valid_loss": valid_loss,
                    'valid_accuracy': valid_accuracy,
                    "valid_F1": valid_F1,
                    
                    "best": best
                })

            print_best = 0    
            if valid_F1 >= best:
                difference = valid_F1 - best
                best = valid_F1 
                best_idx = e+1
                model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.module.state_dict()
                best_model_wts = copy.deepcopy(model_state_dict)
                
                # load and save best model weights
                model.module.load_state_dict(best_model_wts)
                torch.save(model_state_dict, wandb_config['model_save_name']  + '_fold_'+str(fold+1)+ ".pt")
                print_best = '==> best model saved - %d epoch / %.5f    difference %.5f'%(best_idx, best, difference)

            TIME = time.time() - start
            print(f'Fold : {fold+1}/{n_fold}    epoch : {e+1}/{epochs}')
            print(f'TRAIN_1 acc : {train_acc_digit_1:.5f}    TRAIN_2 acc : {train_acc_digit_2:.5f}    TRAIN_3 acc : {train_acc_digit_3:.5f}  |  TRAIN_F1 : {train_F1:.5f}')
            print(f'VALID_1 acc : {valid_acc_digit_1:.5f}    VALID_2 acc : {valid_acc_digit_2:.5f}    VALID_3 acc : {valid_acc_digit_3:.5f}  |  VALID_F1 : {valid_F1:.5f}')
            print(f'TRAIN_acc   : {train_accuracy:.5f}    VALID_acc   : {valid_accuracy:.5f}')
            print(f'VALID_F1 : {valid_F1:.5f}    VALID_Accuracy : {valid_accuracy:.5f}  |  best : {best:.5f}')
            print('\n') if type(print_best)==int else print(print_best,'\n')

            if early_stopping.step(torch.tensor(valid_F1)):
                wandb.join()  
                break
            
        max_index = valid_f1_plot.index(max(valid_f1_plot))
        
        wandb.join()   
        print("Valid Acc: ", valid_acc_plot[max_index], ", Valid Acc: ", valid_f1_plot[max_index])
        print(wandb_config['model_save_name'] + '_fold_ model is saved!')
        torch.cuda.empty_cache() 
        
        k_valid_f1_plot.append(valid_f1_plot[max_index])
        k_valid_acc_plot.append(valid_acc_plot[max_index])
        
    print("k-fold Valid Accuracy: ",np.mean(k_valid_acc_plot),", k-fold Valid F1: ",np.mean(k_valid_f1_plot))

if __name__ == '__main__':
    main()
