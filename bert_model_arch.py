import transformers
from transformers import BertConfig
import torch.nn as nn

class BERTmodel(nn.Module):
    def __init__(self):
        super(BERTmodel, self).__init__()
        self.configuration=BertConfig()
        self.bert=transformers.BertModel(self.configuration)
        self.dropout=nn.Dropout(0.3)
        self.linear=nn.Linear(768,4) #768 is the output size of BERT and 4 is the number of output classes we have

    def forward(self,ip_ids,attention_masks):
        _, op= self.bert(ip_ids, attention_mask = attention_masks,return_dict=False) #op is the embedding of all input tokens and also special tokens. placeholder is only for input tokens.
        op=self.dropout(op)
        op=self.linear(op)
        return op
    
    