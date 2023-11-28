# https://www.nogawanogawa.com/entry/bert_embedding
# https://github.com/yagays/pytorch_bert_japanese/blob/master/bert_juman.py#L46


from transformers import BertJapaneseTokenizer, BertForSequenceClassification, BertForNextSentencePrediction
import pandas as pd
import numpy as np
import torch


class FeaturePretrainedBert:
    
    def __init__(self):
    
        super().__init__()
        self.tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-v3")
        self.model = BertForNextSentencePrediction.from_pretrained(
            "cl-tohoku/bert-base-japanese-v3"
        )
        self.model.eval()


    def get_embedding(self, text: str):
    
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])

        with torch.no_grad(): # 勾配計算なし
    
            all_encoder_layers = self.model(tokens_tensor)

        # embedding = all_encoder_layers[1][-2].numpy()[0]
        # result = np.mean(embedding, axis=0)
        
        embedding = all_encoder_layers[-2].cpu().numpy()[0]
        result = embedding[0]

        return result


    def get(self, primary_text: str, secondary_text: str):
    
        text = primary_text + "[SEP]" + secondary_text
        f8 = self.get_embedding(text)

        return f8


# xInstance = FeaturePretrainedBert()
# x = xInstance.get("吾輩は猫である。", "猫なのである。")
# print(x)

tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-v3")
model = BertForNextSentencePrediction.from_pretrained("cl-tohoku/bert-base-japanese-v3")

encoding = tokenizer("これは何ですか", "助けて", return_tensors="pt")

outputs = model(**encoding, labels=torch.LongTensor([1]))
logits = outputs.logits
is_next = logits[0, 0] > logits[0, 1]
print(is_next)



