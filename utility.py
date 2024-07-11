from transformers import BertTokenizer

def preprocess(sentence):
    tokenizer=BertTokenizer.from_pretrained('bert-base-cased')
    encoding=tokenizer.encode_plus(
            sentence,
            add_special_tokens= True,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
    return encoding

actions={0:'add node',1:'delete node',2:'add property',3:'add relationship'}