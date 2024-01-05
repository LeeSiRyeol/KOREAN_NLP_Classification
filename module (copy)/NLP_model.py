import torch
from torch import nn

#Classifie head usig Dense layer
class Dense_head(nn.Module):
    """
    Input: encoder_output (It is using the [CLS] position output, the [CLS] position is 0th token)
    * encoder_output = output of the hidden layers (base-model using 12 layers)
    Just using last hidden layer
    last_hidden_layer = [batch_size X input_sequence_length X hidden_layer_size(768)]
    
    Basic layer is used Dense network
    
    The number of the classification layer output is the label_num
    
    'logits' represents the prediction score for each class (ranging from min: 0 to max: 1), as the last layer employs a softmax function.
    
    'y_hat' is determined using argmax; it is the index number corresponding to the highest prediction score among the logits.
    """
    def __init__(self,config,label_num):
        super(Dense_head,self).__init__()
        self.config = config
        self.label_num = label_num
        
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier_head = nn.Linear(self.config.hidden_size,label_num)
        self.softmax = nn.Softmax(dim=1)
        
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, encoder_output):
        outputs = encoder_output.last_hidden_state[:,0,:]
        outputs = self.dropout(outputs)
        outputs = self.dense(outputs)
        outputs = torch.tanh(outputs)
        outputs = self.dropout(outputs)
        outputs = self.classifier_head(outputs)
        logits = self.softmax(outputs)
        y_hat = logits.argmax(-1)
        
        return logits, y_hat
    
    def cal_loss(self, logits, label):
        loss = self.loss_fn(logits, label)
        return loss


class TC(nn.Module):
    def __init__(self,device, head='Dense', backbone='KLUE-RoBERTa',label_num=3):
        super(TC,self).__init__()
        self.device = device
        self.head = head
        self.backbone = backbone
        self.label_num = label_num

        if self.backbone == "KLUE-RoBERTa":
            from transformers import RobertaModel
            self.backbone_model = RobertaModel.from_pretrained("klue/roberta-base",add_pooling_layer=False,output_hidden_states=True)
            self.config = self.backbone_model.config
            
        elif self.backbone == "KLUE-BERT":
            from transformers import AutoModel
            self.backbone_model = AutoModel.from_pretrained("klue/bert-base",add_pooling_layer=False,output_hidden_states=True)
            self.config = self.backbone_model.config
            
        elif self.backbone == "KoBERT":
            from transformers import BertModel
            self.backbone_model = BertModel.from_pretrained("skt/kobert-base-v1",add_pooling_layer=False,output_hidden_states=True)
            self.config = self.backbone_model.config
            
        elif self.backbone == "KoBigBird":
            from transformers import AutoModel
            self.backbone_model = AutoModel.from_pretrained("monologg/kobigbird-bert-base") 
            self.config = self.backbone_model.config
            
        elif self.backbone == "KorBERT":
            from pytorch_pretrained_bert.modeling import BertForSequenceClassification
            self.backbone_model = BertForSequenceClassification.from_pretrained('./module/kor_tensorflow', num_labels=label_num)
            self.config = self.backbone_model.config
            self.head = None
            
        
        if self.head == 'Dense':
            self.classifier_head = Dense_head(self.config, self.label_num)
            
    def __enter__(self):
        print("TC start")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("TC finish")

    def forward(self,input_ids, attention_mask, labels):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)
        
        if self.backbone == "KorBERT":
            if self.training:
                self.backbone_model.train()
                encoder_outputs = self.backbone_model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                self.backbone_model.eval()
                with torch.no_grad():
                    encoder_outputs = self.backbone_model(input_ids=input_ids, attention_mask=attention_mask)
                    
            logits = encoder_outputs
            y_hat = logits.argmax(-1)
            
        else:
            if self.training:
                self.backbone_model.train()
                encoder_outputs = self.backbone_model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                self.backbone_model.eval()
                with torch.no_grad():
                    encoder_outputs = self.backbone_model(input_ids=input_ids, attention_mask=attention_mask)

        if self.head == 'Dense':
            logits, y_hat = self.classifier_head(encoder_outputs)

        return logits, labels, y_hat