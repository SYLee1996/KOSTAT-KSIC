import os 
import re
import copy
import click
import random
import datetime
import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from konlpy.tag import Okt, Kkma, Mecab
from gensim.models import Word2Vec

import warnings
warnings.filterwarnings(action='ignore')

@click.command()
@click.option('--save_name', type=click.STRING, required=True)
@click.option('--tokenizer_name', type=click.STRING, default='Mecab', required=True)
@click.option('--embedding_iter', type=click.STRING, default='5', required=True)

def main(save_name, tokenizer_name, embedding_iter):
    
    seed = 10
    
    suffix = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime("%y%m%d_%H%M")
    save_name = suffix+'_'+save_name           
    Model_library = eval(tokenizer_name)
    embedding_iter = int(embedding_iter)

    # ---------------------------------------------------------------------------------------------
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 데이터 로드-----------------------------------------------------------------------------------
    train = pd.read_csv('1. 실습용자료.txt', sep = "|", engine='python', encoding='cp949')
    test = pd.read_csv('2. 모델개발용자료.txt', sep = "|", engine='python', encoding='cp949')
    label_data_raw = pd.read_excel('한국표준산업분류(10차)_국문.xlsx',header=[0,1,2])
    label_data_raw = label_data_raw.fillna(method='ffill')
    label_data = label_data_raw['개정 분류체계(제10차 기준)'][['대분류(21)', '중분류(77)', '소분류(232)']]

    digit_1 = list(label_data['대분류(21)']['코드'].dropna().unique())
    digit_2 = list(label_data['중분류(77)']['코드'].dropna().unique())
    digit_2 = list(map(int, digit_2))
    digit_3 = list(label_data['소분류(232)']['코드'].dropna().unique())
    digit_3 = list(map(int, digit_3))


    # train 데이터 전처리 --------------------------------------------------------------------------
    train['text_obj'] = train['text_obj'].str.replace('[^가-힣 0-9 a-z A-Z]+$', '', regex=True)
    train['text_mthd'] = train['text_mthd'].str.replace('[^가-힣 0-9 a-z A-Z]+$', '', regex=True)
    train['text_deal'] = train['text_deal'].str.replace('[^가-힣 0-9 a-z A-Z]+$', '', regex=True)
    train[['text_obj', 'text_mthd', 'text_deal']] = train[['text_obj', 'text_mthd', 'text_deal']].applymap(lambda x: x.strip() if isinstance(x, str) else x)

    train['text_obj'].replace(np.nan, '', inplace=True)
    train['text_obj'].replace(' ', '', inplace=True)

    train['text_mthd'].replace(np.nan, '', inplace=True)
    train['text_mthd'].replace(' ', '', inplace=True)

    train['text_deal'].replace(np.nan, '', inplace=True)
    train['text_deal'].replace(' ', '', inplace=True)

    train['text_deal'].iloc[436664] = '석탄 채굴'

    add_dict = {'AI_id':['id_1000001', 'id_1000002', 'id_1000003', 'id_1000004', 'id_1000005', 'id_1000006', 'id_1000007', 'id_1000008'],
            'digit_1': ['A', 'A', 'B', 'T', 'T', 'T', 'U', 'U'],
            'digit_2': [1, 1, 5, 97, 98, 98, 99, 99],
            'digit_3': [13, 15, 52, 970, 981, 982, 990, 990],
            'text_obj': ['', '', '', '', '', '', '', ''],
            'text_mthd': ['', '', '', '', '', '', '', ''],
            'text_deal': ['작물재배 및 축산 복합농업', '수렵 및 관련 서비스업', '원유 및 천연가스 채굴업', '가구 내 고용활동', '자가 소비를 위한 가사 생산 활동', '자가 소비를 위한 가사 서비스 활동', '주한 외국 공관', '국제 및 외국기관']}
    add_df = pd.DataFrame(add_dict)
    train = pd.concat([train, add_df])

    train['concat'] = train['text_obj'] + ' ' + train['text_mthd'] + ' ' +  train['text_deal']
    train['concat'] = train['concat'].apply(lambda x: x.strip())
    
    # label encoding
    digit_1_encoder = LabelEncoder()
    digit_1_encoder.fit(digit_1)

    digit_2_encoder = LabelEncoder()
    digit_2_encoder.fit(digit_2)

    digit_3_encoder = LabelEncoder()
    digit_3_encoder.fit(digit_3)

    train['digit_1_encoded'] = digit_1_encoder.transform(train['digit_1'])
    train['digit_2_encoded'] = digit_2_encoder.transform(train['digit_2'])
    train['digit_3_encoded'] = digit_3_encoder.transform(train['digit_3'])
    
    # -------------------------------------------

    digit_1_code = label_data_raw['개정 분류체계(제10차 기준)']['대분류(21)']['코드']
    digit_2_code = label_data_raw['개정 분류체계(제10차 기준)']['중분류(77)']['코드'] 
    digit_3_code = label_data_raw['개정 분류체계(제10차 기준)']['소분류(232)']['코드']

    AUG_DATA = pd.concat([digit_1_code, digit_2_code, digit_3_code], axis=1)
    AUG_DATA.columns = ['digit_1','digit_2','digit_3']
    AUG_DATA['digit_2'] = AUG_DATA['digit_2'].map(int)
    AUG_DATA['digit_3'] = AUG_DATA['digit_3'].map(int)

    AUG_DATA_small = copy.copy(AUG_DATA)
    AUG_DATA_se = copy.copy(AUG_DATA)
    AUG_DATA_sese = copy.copy(AUG_DATA)

    AUG_DATA_small['text_obj'] = ''	
    AUG_DATA_small['text_mthd'] = ''	
    AUG_DATA_small['text_deal'] = label_data_raw['개정 분류체계(제10차 기준)']['소분류(232)']['항목명']

    AUG_DATA_se['text_obj'] = ''	
    AUG_DATA_se['text_mthd'] = ''	
    AUG_DATA_se['text_deal'] = label_data_raw['개정 분류체계(제10차 기준)']['세분류(495)']['항목명']

    AUG_DATA_sese['text_obj'] = ''	
    AUG_DATA_sese['text_mthd'] = ''	
    AUG_DATA_sese['text_deal'] = label_data_raw['개정 분류체계(제10차 기준)']['세세분류(1,196)']['항목명']

    # 개정 분류체계(제10차 기준) 기반 데이터 증각의 경우 한 단어로만 표현되는 데이터는 지우지 않음 

    AUG_DATA_small['digit_1_encoded'] = digit_1_encoder.transform(AUG_DATA_small['digit_1'])
    AUG_DATA_small['digit_2_encoded'] = digit_2_encoder.transform(AUG_DATA_small['digit_2'])
    AUG_DATA_small['digit_3_encoded'] = digit_3_encoder.transform(AUG_DATA_small['digit_3'])

    AUG_DATA_se['digit_1_encoded'] = digit_1_encoder.transform(AUG_DATA_se['digit_1'])
    AUG_DATA_se['digit_2_encoded'] = digit_2_encoder.transform(AUG_DATA_se['digit_2'])
    AUG_DATA_se['digit_3_encoded'] = digit_3_encoder.transform(AUG_DATA_se['digit_3'])

    AUG_DATA_sese['digit_1_encoded'] = digit_1_encoder.transform(AUG_DATA_sese['digit_1'])
    AUG_DATA_sese['digit_2_encoded'] = digit_2_encoder.transform(AUG_DATA_sese['digit_2'])
    AUG_DATA_sese['digit_3_encoded'] = digit_3_encoder.transform(AUG_DATA_sese['digit_3'])

    to_token_df1 = label_data_raw['개정 분류체계(제10차 기준)']['소분류(232)']['항목명'].drop_duplicates()
    to_token_df2 = label_data_raw['개정 분류체계(제10차 기준)']['세분류(495)']['항목명'].drop_duplicates()
    to_token_df3 = label_data_raw['개정 분류체계(제10차 기준)']['세세분류(1,196)']['항목명'].drop_duplicates()
    
    
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

    train['text_obj_stop'] = text_preprocessing(train['text_obj']) 
    train['text_deal_stop'] = text_preprocessing(train['text_deal']) 
    train['text_mthd_stop'] = text_preprocessing(train['text_mthd']) 
    train['token'] = text_preprocessing(train['concat']) 
    train.drop(train[train['token'].map(len) == 1].index, inplace=True)
    train = train.reset_index(drop=True)
    
    AUG_DATA_small['token'] = text_preprocessing(AUG_DATA_small['text_deal']) 
    AUG_DATA_se['token'] = text_preprocessing(AUG_DATA_se['text_deal']) 
    AUG_DATA_sese['token'] = text_preprocessing(AUG_DATA_sese['text_deal']) 

    to_token_list1 = text_preprocessing(to_token_df1) 
    to_token_list2 = text_preprocessing(to_token_df2) 
    to_token_list3 = text_preprocessing(to_token_df3) 
    
    #  ------------------------------------------------------------------------------------------
    train_save = pd.concat([train[['digit_1_encoded', 'digit_2_encoded', 'digit_3_encoded', 'token']],
                            AUG_DATA_small[['digit_1_encoded', 'digit_2_encoded', 'digit_3_encoded', 'token']],
                            AUG_DATA_se[['digit_1_encoded', 'digit_2_encoded', 'digit_3_encoded', 'token']],
                            AUG_DATA_sese[['digit_1_encoded', 'digit_2_encoded', 'digit_3_encoded', 'token']]])
    train_save = train_save.reset_index(drop=True)
    
    np.save(save_name+'.npy', train_save)
    print(save_name+'.npy is saved!')
    
    # word2vec ----------------------------------------------------------------------------------
    train_token_list = list(train['text_obj_stop']) + list(train['text_mthd_stop']) + list(train['text_deal_stop'])\
        + to_token_list1 + to_token_list2 + to_token_list3

    model_pretrained = Word2Vec.load('ko.bin')
    model_CBOW_train = Word2Vec(
                            sg=0,      
                            size=model_pretrained.vector_size,    
                            window=3,    
                            min_count=1, 
                            workers=4, seed=10) 
    model_CBOW_train.build_vocab([model_pretrained.wv.vocab.keys()])
    model_CBOW_train.build_vocab(train_token_list, update=True)
    model_CBOW_train.train(train_token_list, total_examples=len(train_token_list), epochs=embedding_iter)
    
    words = ['석탄', '원유', '채굴', '담배', '생산']
    for word in words:
        print('{}:  '.format(word), model_CBOW_train.wv.most_similar(word)[:5])
    
    # vocab & embedding save --------------------------------------------------------------------
    vocab = list(model_CBOW_train.wv.vocab) # Word2Vec에서 사용한 vocab
    trained_vectors=[]
    for vo in vocab:
        trained_vectors.append(model_CBOW_train[vo])
        
    vocab_npa = np.array(vocab)
    embs_npa = np.array(trained_vectors)

    #insert '<pad>' and '<unk>' tokens at start of vocab_npa.
    vocab_npa = np.insert(vocab_npa, 0, '<pad>')
    vocab_npa = np.insert(vocab_npa, 1, '<unk>')
    print('vocab_npa[:10]: ', vocab_npa[:10])

    pad_emb_npa = np.zeros((1,embs_npa.shape[1]))   #embedding for '<pad>' token.
    unk_emb_npa = np.mean(embs_npa,axis=0,keepdims=True)    #embedding for '<unk>' token.

    #insert embeddings for pad and unk tokens at top of embs_npa.
    embs_npa = np.vstack((pad_emb_npa, unk_emb_npa, embs_npa))
    print('embs_npa shape: ', embs_npa.shape)
    
    with open(save_name+'_vocab_npa_{}.npy'.format(embedding_iter),'wb') as f:
        np.save(f,vocab_npa)

    with open(save_name+'_embs_npa_{}.npy'.format(embedding_iter),'wb') as f:
        np.save(f,embs_npa)
        
        
    print(save_name+'_vocab_npa_{}.npy'.format(embedding_iter) + ' is Saved!')
    print(save_name+'_embs_npa_{}.npy'.format(embedding_iter) + ' is Saved!')
    
if __name__ == '__main__':
    main()
