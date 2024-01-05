import pandas as pd
from torch.utils import data
import torch

def Data_Load(path):
    load_data = pd.read_csv(path)
    
    load_text = pd.DataFrame(load_data['text'])
    load_class = pd.DataFrame(load_data['class'])
    
    return load_text, load_class

class DATA(data.Dataset):
    def __init__(self, txt, label, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.txt = txt
        self.label = label
        self.seq_len = seq_len

    def __len__(self):
        return len(self.txt)

    def __getitem__(self,idx):
        text = self.txt['text'][idx]
        
        label = self.label['class'][idx]
        
        token = self.tokenizer(text)
        
        
        if len(token.input_ids) > self.seq_len:
            input_ids = token.input_ids[:self.seq_len]
            token_type_ids = token.token_type_ids[:self.seq_len]
            attention_mask = token.attention_mask[:self.seq_len]
        
        else:
            input_ids = token.input_ids
            token_type_ids = token.token_type_ids
            attention_mask = token.attention_mask
        
        return input_ids, attention_mask, token_type_ids, label
        
    
    def pad(self,batch):
        temp = lambda x: [sample[x] for sample in batch]
        # seq_len = [len(sample[0]) for sample in batch]
        
        input_ids = temp(0)
        attention_mask = temp(1)
        token_type_ids = temp(2)
        label = temp(3)
        max_len = self.seq_len#np.array(seq_len).max()
        
        padding = lambda x, value, seqlen: torch.tensor([sample + [value] * (seqlen - len(sample)) for sample in x], dtype=torch.int64)
        
        input_ids = padding(input_ids, self.tokenizer.pad_token_id, max_len)
        attention_mask = padding(attention_mask, 0, max_len)
        token_type_ids = padding(token_type_ids, self.tokenizer.pad_token_type_id, max_len)
        label = torch.tensor(label,dtype=torch.int64)
        
        return input_ids, attention_mask, token_type_ids, label   
    
    
class DATA_kor(data.Dataset):
    def __init__(self, txt, label, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.txt = txt
        self.label = label
        self.seq_len = seq_len

    def __len__(self):
        return len(self.txt)

    def __getitem__(self,idx):
        text = self.txt['text'][idx]
        
        label = self.label['class'][idx]
        
        token = self.tokenizer.tokenize(text)
        
        input_ids = self.tokenizer.convert_tokens_to_ids(token)
        token_type_ids = [0]*len(input_ids)
        attention_mask = [1]*len(input_ids)
        
        
        if len(input_ids) > self.seq_len:
            input_ids = input_ids[:self.seq_len]
            token_type_ids = token_type_ids[:self.seq_len]
            attention_mask = attention_mask[:self.seq_len]
        
        return input_ids, attention_mask, token_type_ids, label
        
    
    def pad(self,batch):
        temp = lambda x: [sample[x] for sample in batch]
        # seq_len = [len(sample[0]) for sample in batch]
        
        input_ids = temp(0)
        attention_mask = temp(1)
        token_type_ids = temp(2)
        label = temp(3)
        max_len = self.seq_len#np.array(seq_len).max()
        
        padding = lambda x, value, seqlen: torch.tensor([sample + [value] * (seqlen - len(sample)) for sample in x], dtype=torch.int64)
        
        input_ids = padding(input_ids, 0, max_len)
        attention_mask = padding(attention_mask, 0, max_len)
        token_type_ids = padding(token_type_ids, 0, max_len)
        label = torch.tensor(label,dtype=torch.int64)
        
        return input_ids, attention_mask, token_type_ids, label   