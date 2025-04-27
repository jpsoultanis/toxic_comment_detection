# inference.py

import torch
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from data_cleaning import clean_text


def prepare_test_data(df, vocab, max_len):
    df['comment_text'] = df['comment_text'].apply(clean_text)
    df['tokens'] = df['comment_text'].apply(word_tokenize)

    def encode(tokens):
        return [vocab.get(tok, vocab['<UNK>']) for tok in tokens[:max_len]]

    def pad(seq):
        return seq + [0] * (max_len - len(seq))

    df['input_ids'] = df['tokens'].apply(lambda x: pad(encode(x)))
    return df


def run_inference(model, test_loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, _ in test_loader:  # labels are dummy here
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            predictions.extend(preds)

    return np.array(predictions)
