import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import os
import random

from sklearn.metrics import f1_score, classification_report

import torch
from torch import nn
from torch.utils import data
import torch.optim as optim

from module.dataload import Data_Load
from module.NLP_model import TC

class Processing(nn.Module):
    def __init__(self, test_batch, label_list):
        self.test_batch = test_batch
        self.label_list = label_list
        self.F1_LIST = list()
    def __enter__(self):
        print("Evaluation Start")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Evaluation Exit")

    def Test(self,model):
        true_labels = []
        predicted_labels = []
        logits_list = []
        with torch.no_grad():
            print('---------------Test Data Prediction Start---------------')
            model.eval()
            for i, batch in enumerate(self.test_batch):
                input_ids, attention_mask, token_type_ids, labels = batch
                logits, _, y_hat = model(input_ids,attention_mask, labels)
                    
                _logits = logits.cpu().numpy()
                _y_hat = y_hat.cpu().numpy()
                _labels = labels.cpu().numpy()
                    
                logits_list.extend(_logits)
                true_labels.extend(np.take(self.label_list,_labels).tolist()) 
                predicted_labels.extend(np.take(self.label_list,_y_hat).tolist())

        f1 = f1_score(true_labels, predicted_labels, average=None)
        self.F1_LIST.append(f1)
        self.REPORT = classification_report(true_labels, predicted_labels)

        self.PREDICTED_LABELS = predicted_labels
        self.TRUE_LABELS = true_labels
        self.LOGIT_LIST = logits_list
        print('---------------Test Data Prediction Finish---------------')
            
    
    def test_result(self):
        return self.F1_LIST, self.REPORT, self.PREDICTED_LABELS, self.TRUE_LABELS, self.LOGIT_LIST



if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--NLP_model", type=str, default='KLUE-RoBERTa')
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--random_seed", type=int, default=102)
    parser.add_argument("--data_path", type=str, default='./data')  
    args = parser.parse_args()
    
    batch_size = args.batch_size
    NLP_model = args.NLP_model
    seq_len = args.seq_len
    data_path = args.data_path
    random_seed = args.random_seed
    
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    if NLP_model == 'KLUE-RoBERTa':
        from module.dataload import DATA
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base",TOKENIZERS_PARALLELISM=True)
        
    elif NLP_model == 'KLUE-BERT':
        from module.dataload import DATA
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("klue/bert-base",TOKENIZERS_PARALLELISM=True)
        
    elif NLP_model == 'KoBERT':
        from module.dataload import DATA
        from kobert_tokenizer import KoBERTTokenizer
        tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1",TOKENIZERS_PARALLELISM=True)
        
    elif NLP_model == 'KoBigBird':
        from module.dataload import DATA
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base",TOKENIZERS_PARALLELISM=True)
        seq_len = 1024
        
    elif NLP_model == 'KorBERT':
        from module.dataload import DATA_kor as DATA
        from module.kor_tensorflow.src_tokenizer import tokenization
        from module.kor_tensorflow.src_tokenizer.tokenization import BertTokenizer
        tokenizer_path = os.path.join('./module/kor_tensorflow', 'vocab.korean.rawtext.list')
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)


    # test_txt, test_class
    test_txt, test_class = Data_Load(data_path+'/example_data.csv') #  +'/test.csv
    
    test_data = DATA(test_txt, test_class, tokenizer, seq_len)

    test_batch = data.DataLoader(dataset=test_data,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=4,
                                  collate_fn=test_data.pad)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    label_list = ['False','True']
    
    with TC(head='Dense', backbone=NLP_model, device=device, label_num=len(label_list)).cuda() as model:    
        model_path = "./trained_model.txt"
        model.load_state_dict(torch.load(model_path))
        
        with Processing(test_batch,label_list) as test:            
            test.Test(model)
            F1_LIST, REPORT, PREDICTED_LABELS, TRUE_LABELS, LOGIT_LIST = test.test_result() 
            