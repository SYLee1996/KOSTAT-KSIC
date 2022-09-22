import os 
import re
import copy 
import click
import random
import datetime
import collections
import numpy as np
import pandas as pd
from konlpy.tag import Okt, Kkma, Mecab
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from INDUSTRY_CODE_MODEL import LSTMEncoder
from INDUSTRY_CODE_UTILS import NLP_Dataset, predict, label_mask

import warnings
warnings.filterwarnings(action='ignore')

@click.command()
@click.option('--model_name', type=click.STRING, required=True)
@click.option('--device', type=click.STRING, default='0,1,2,3', required=True)
@click.option('--vocab_load', type=click.STRING, required=True)
@click.option('--embedd_load', type=click.STRING, required=True)
@click.option('--n_fold', type=click.STRING, default=5, required=True)
@click.option('--hidden_size', type=click.STRING, required=True)
@click.option('--lstm_unit_cnt', type=click.STRING, required=True)
@click.option('--max_seq_len', type=click.STRING, default=256, required=True)
@click.option('--batch_size', type=click.STRING, default=16,required=True)
@click.option('--tokenizer_name', type=click.STRING, default='Mecab', required=True)

def main(model_name, vocab_load, embedd_load, n_fold, hidden_size, lstm_unit_cnt, device, max_seq_len, batch_size, tokenizer_name):
    
    seed = 10    
    vocab_npa = np.load(vocab_load)
    embs_npa = np.load(embedd_load)
    Model_library = eval(tokenizer_name)
    
    config = {
        'n_fold': int(n_fold),
        'freeze_embeddings': False,
        'hidden_size': int(hidden_size),
        'lstm_unit_cnt': int(lstm_unit_cnt),
        'max_seq_len': int(max_seq_len),
        'dropout':0.0,
        'batch_size':int(batch_size),
        'device': device,
        
        'model_name': model_name,
        
        'vocab': vocab_npa,
        'pretrained_embeddings': embs_npa,
        'PAD_TOK':'<pad>',
        'UNK_TOK':'<unk>'
    }

    # -------------------------------------------------------------------------------------------
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"]=config['device']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    print('Device: %s' % device)
    if (device.type == 'cuda') or (torch.cuda.device_count() > 1):
        print('GPU activate --> Count of using GPUs: %s' % torch.cuda.device_count())
    config['model_device'] = device
        
    # -------------------------------------------------------------------------------------------
    
    test = pd.read_csv('2. 모델개발용자료.txt', sep = "|", engine='python', encoding='cp949')
    label_data = pd.read_excel('한국표준산업분류(10차)_국문.xlsx',header=[0,1,2])
    label_data = label_data.fillna(method='ffill')
    label_data = label_data['개정 분류체계(제10차 기준)'][['대분류(21)', '중분류(77)', '소분류(232)']]

    digit_1 = list(label_data['대분류(21)']['코드'].dropna().unique())
    digit_2 = list(label_data['중분류(77)']['코드'].dropna().unique())
    digit_2 = list(map(int, digit_2))
    digit_3 = list(label_data['소분류(232)']['코드'].dropna().unique())
    digit_3 = list(map(int, digit_3))
    
    config['digit_1_num'] = len(digit_1)
    config['digit_2_num'] = len(digit_2)
    config['digit_3_num'] = len(digit_3)
    
    # test 데이터 전처리
    test['text_obj'] = test['text_obj'].str.replace('[^가-힣 0-9 a-z A-Z]+$', '', regex=True)
    test['text_mthd'] = test['text_mthd'].str.replace('[^가-힣 0-9 a-z A-Z]+$', '', regex=True)
    test['text_deal'] = test['text_deal'].str.replace('[^가-힣 0-9 a-z A-Z]+$', '', regex=True)
    test[['text_obj', 'text_mthd', 'text_deal']] = test[['text_obj', 'text_mthd', 'text_deal']].applymap(lambda x: x.strip() if isinstance(x, str) else x)

    test['text_obj'].replace(np.nan, '', inplace=True)
    test['text_obj'].replace(' ', '', inplace=True)

    test['text_mthd'].replace(np.nan, '', inplace=True)
    test['text_mthd'].replace(' ', '', inplace=True)

    test['text_deal'].replace(np.nan, '', inplace=True)
    test['text_deal'].replace(' ', '', inplace=True)
    test['concat'] = test['text_obj'] + ' ' + test['text_mthd'] + ' ' +  test['text_deal']
    test['concat'] = test['concat'].apply(lambda x: x.strip())
    test = test.reset_index(drop=True)

    # label encoding

    digit_1_encoder = LabelEncoder()
    digit_1_encoder.fit(digit_1)

    digit_2_encoder = LabelEncoder()
    digit_2_encoder.fit(digit_2)

    digit_3_encoder = LabelEncoder()
    digit_3_encoder.fit(digit_3)
    
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

    # 불용어 제거 ----------------------------------------------------------------------------------
    
    def text_preprocessing(text_list):
            
        stopwords = ['을', '를', '이', '가', 
                    '은', '는', '고', '그',
                    '의', '및','등', '외',
                    '와','한다','하','에',
                    '않', '안', '된'] # 불용어 설정

        tokenizer = Model_library() # 형태소 분석기 
        token_list = []
        
        for text in text_list.values:
            txt = re.sub('[^가-힣 0-9 a-z A-Z]', '', text) # 한글과 영어 소문자만 남기고 다른 글자 모두 제거
            li = []
            for word in txt.split(" "):
                token = tokenizer.morphs(word)
                li += token
            token = [t for t in li if t not in stopwords and type(t) != float] # 형태소 분석 결과 중 stopwords에 해당하지 않는 것만 추출
            token_list.append(token)
            
        return token_list

    test['token'] = text_preprocessing(test['concat']) 
    
    # -------------------------------------------------------------------------------------------
    
    # Test
    models = []
    for fold in range(config['n_fold']):
        model = LSTMEncoder(config, config).to(device) 
        model = nn.DataParallel(model).to(device)
        model_dict = torch.load('./RESULTS/'+config['model_name'] + str(fold+1) + ".pt")
        model.module.load_state_dict(model_dict) if torch.cuda.device_count() > 1 else model.load_state_dict(model_dict)
        models.append(model)
            
    # Test 
    test_dataset = NLP_Dataset(test, 
                                config['vocab'],
                                config['max_seq_len'], config['PAD_TOK'],
                                config['UNK_TOK'], train=False)
    
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                            num_workers=16, prefetch_factor=config['batch_size']*2, 
                            drop_last=False, shuffle=False, pin_memory=True)

    pred_1,pred_2,pred_3 = predict(models, test_loader, config)

    sub = test[['AI_id','digit_1','digit_2','digit_3','text_obj','text_mthd','text_deal']]
    sub['digit_1'] = digit_1_encoder.inverse_transform(pred_1)
    sub['digit_2'] = digit_2_encoder.inverse_transform(pred_2)
    sub['digit_3'] = digit_3_encoder.inverse_transform(pred_3)

    sub.to_csv("./RESULTS/{}.csv".format(config['model_name']), index=False)
    print(config['model_name'] + " is saved!")
    
if __name__ == '__main__':
    main()
