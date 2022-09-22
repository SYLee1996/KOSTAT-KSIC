import copy 
import torch
from torch import nn
from functools import partial
from torch.multiprocessing import Pool

class LSTMEncoder(torch.nn.Module):
    def __init__(self, config, config_origin):
        super(LSTMEncoder, self).__init__()
        
        pretrained_embeddings = config_origin['pretrained_embeddings']
        freeze_embeddings = config['freeze_embeddings']
        
        if pretrained_embeddings is not None:
            self.vocab_size = pretrained_embeddings.shape[0]
            self.embedding_dim = pretrained_embeddings.shape[1]
            self.embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(pretrained_embeddings).float(),
                                                                freeze=freeze_embeddings)
        
        self.hidden_size = config['hidden_size']
        self.lstm_unit_cnt = config['lstm_unit_cnt']
        self.dropout = config['dropout'] 
        self.device = config_origin['model_device']
        self.lstm = torch.nn.LSTM(input_size=self.embedding_dim,
                                    hidden_size=self.hidden_size,
                                    num_layers=self.lstm_unit_cnt,
                                    batch_first=True,
                                    bidirectional=False,
                                    dropout=self.dropout)
        
        self.digit_1 = nn.Linear(self.hidden_size, config['digit_1_num'])
        self.digit_2 = nn.Linear(self.hidden_size, config['digit_2_num'])
        self.digit_3 = nn.Linear(self.hidden_size, config['digit_3_num'])
    
        self.digit_1_to_2_dict = config_origin['digit_1_to_2_mask']
        self.digit_2_to_3_dict = config_origin['digit_2_to_3_mask']
        
    def forward(self, batch, valid=False):
        x = batch['input_ids']
        x_lengths = batch['sequence_len']
        embed_out = self.embedding(x)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(embed_out, x_lengths.tolist(),
                                                                enforce_sorted=False,
                                                                batch_first=True)
        packed_out,_ = self.lstm(packed_input)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)    #inverse operation of pack_padded_sequence
        
        lstm_out = output[range(len(output)), x_lengths - 1, :self.hidden_size]
        
        x_out_1 = self.digit_1(lstm_out)
        x_out_2 = self.digit_2(lstm_out)
        x_out_3 = self.digit_3(lstm_out)
        
        if valid == False:
            return x_out_1, x_out_2, x_out_3
        
        elif valid == True:
            _, max_indices_1 = torch.max(x_out_1, 1)
            with Pool(1) as pool:
                mask_1 = pool.map(partial(masking, digit_mask=self.digit_1_to_2_dict), max_indices_1.tolist())
            mask_1 = torch.Tensor(mask_1).bool().to(self.device)
            x_out_2 = x_out_2.masked_fill_(mask_1, -10000.)
            
            _, max_indices_2 = torch.max(x_out_2, 1)
            with Pool(1) as pool:
                mask_2 = pool.map(partial(masking, digit_mask=self.digit_2_to_3_dict), max_indices_2.tolist())
            mask_2 = torch.Tensor(mask_2).bool().to(self.device)
            x_out_3 = x_out_3.masked_fill_(mask_2, -10000.)
            
            return x_out_1, x_out_2, x_out_3
    
def masking(x, digit_mask):
    return digit_mask[x]