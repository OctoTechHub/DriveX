from fastai.text.all import TextDataLoaders, Learner, accuracy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch.nn as nn
import torch

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, output, target):
        logits = output.logits 
        return self.loss_fn(logits, target)

def custom_accuracy(preds, targets):
    preds = preds.logits.argmax(dim=-1) 
    return (preds == targets).float().mean()

if __name__ == '__main__':
    df = pd.read_csv('file_data.csv')

    dls = TextDataLoaders.from_df(df, text_col='text', label_col='label', valid_pct=0.2, bs=32)

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=dls.c)

    learn = Learner(dls, model, loss_func=CustomCrossEntropyLoss(), metrics=custom_accuracy)

    learn.fine_tune(4)

    learn.export('file_classifier.pkl')


    from pathlib import Path
    Path('run.txt').write_text('Script executed successfully')

