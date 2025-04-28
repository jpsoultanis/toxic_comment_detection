# dataset.py
import re
import nltk
import numpy as np
import pandas as pd
import torch
import random
import emoji
import os
import zipfile
import requests

from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

nltk.download('punkt')
nltk.download('punkt_tab') # Download punkt_tab resource

patterns = [
    r'\\[nrtbfv\\]',         # \n, \t ..etc
    '<.*?>',                 # Html tags
    r'https?://\S+|www\.\S+',# Links
    r'\ufeff',               # BOM characters
    r'^[^a-zA-Z0-9]+$',      # Non-alphanumeric tokens
    r'ｗｗｗ．\S+',            # Full-width URLs
    r'[\uf700-\uf7ff]',      # Unicode private-use chars
    r'^[－—…]+$',            # Special punctuation-only tokens
    r'[︵︶]'                # CJK parentheses
]

chat_words = {
    "AFAIK": "As Far As I Know",
    "AFK": "Away From Keyboard",
    "ASAP": "As Soon As Possible",
    "ATK": "At The Keyboard",
    "ATM": "At The Moment",
    "A3": "Anytime, Anywhere, Anyplace",
    "BAK": "Back At Keyboard",
    "BBL": "Be Back Later",
    "BBS": "Be Back Soon",
    "BFN": "Bye For Now",
    "B4N": "Bye For Now",
    "BRB": "Be Right Back",
    "BRT": "Be Right There",
    "BTW": "By The Way",
    "B4": "Before",
    "B4N": "Bye For Now",
    "CU": "See You",
    "CUL8R": "See You Later",
    "CYA": "See You",
    "FAQ": "Frequently Asked Questions",
    "FC": "Fingers Crossed",
    "FWIW": "For What It's Worth",
    "FYI": "For Your Information",
    "GAL": "Get A Life",
    "GG": "Good Game",
    "GN": "Good Night",
    "GMTA": "Great Minds Think Alike",
    "GR8": "Great!",
    "G9": "Genius",
    "IC": "I See",
    "ICQ": "I Seek you (also a chat program)",
    "ILU": "ILU: I Love You",
    "IMHO": "In My Honest/Humble Opinion",
    "IMO": "In My Opinion",
    "IOW": "In Other Words",
    "IRL": "In Real Life",
    "KISS": "Keep It Simple, Stupid",
    "LDR": "Long Distance Relationship",
    "LMAO": "Laugh My A.. Off",
    "LOL": "Laughing Out Loud",
    "LTNS": "Long Time No See",
    "L8R": "Later",
    "MTE": "My Thoughts Exactly",
    "M8": "Mate",
    "NRN": "No Reply Necessary",
    "OIC": "Oh I See",
    "PITA": "Pain In The A..",
    "PRT": "Party",
    "PRW": "Parents Are Watching",
    "QPSA?": "Que Pasa?",
    "ROFL": "Rolling On The Floor Laughing",
    "ROFLOL": "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO": "Rolling On The Floor Laughing My A.. Off",
    "SK8": "Skate",
    "STATS": "Your sex and age",
    "ASL": "Age, Sex, Location",
    "THX": "Thank You",
    "TTFN": "Ta-Ta For Now!",
    "TTYL": "Talk To You Later",
    "U": "You",
    "U2": "You Too",
    "U4E": "Yours For Ever",
    "WB": "Welcome Back",
    "WTF": "What The F...",
    "WTG": "Way To Go!",
    "WUF": "Where Are You From?",
    "W8": "Wait...",
    "7K": "Sick:-D Laugher",
    "TFW": "That feeling when",
    "MFW": "My face when",
    "MRW": "My reaction when",
    "IFYP": "I feel your pain",
    "TNTL": "Trying not to laugh",
    "JK": "Just kidding",
    "IDC": "I don't care",
    "ILY": "I love you",
    "IMU": "I miss you",
    "ADIH": "Another day in hell",
    "ZZZ": "Sleeping, bored, tired",
    "WYWH": "Wish you were here",
    "TIME": "Tears in my eyes",
    "BAE": "Before anyone else",
    "FIMH": "Forever in my heart",
    "BSAAW": "Big smile and a wink",
    "BWL": "Bursting with laughter",
    "BFF": "Best friends forever",
    "CSL": "Can't stop laughing"
}


def clean_text(text):
  for regex in patterns:
    text = re.sub(regex, '', text)
  text = ' '.join(chat_words.get(word.lower(), word) for word in text.split())
  text = text.lower()
  text = emoji.demojize(text)
  text = re.sub(r'\s+', ' ', text).strip()
  return text

def preprocess_and_tokenize(df, max_len):
    df['comment_text'] = df['comment_text'].apply(clean_text)
    df['tokens'] = df['comment_text'].apply(word_tokenize)

    all_tokens = [tok for toks in df['tokens'] for tok in toks]
    vocab = {word: i+2 for i, (word, _) in enumerate(Counter(all_tokens).items())}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1

    def encode(tokens):
        return [vocab.get(token, vocab['<UNK>']) for token in tokens[:max_len]]

    def pad(seq):
        return seq + [0] * (max_len - len(seq))

    df['input_ids'] = df['tokens'].apply(lambda t: pad(encode(t)))
    return df, vocab


def load_glove_embeddings(glove_path, vocab, embedding_dim):
    """
    Load GloVe embeddings from a local file, downloading and extracting if necessary.
    Args:
        glove_path (str): Path to the GloVe .txt file (e.g., 'glove.6B.100d.txt').
        vocab (dict): Mapping from word to index in the embedding matrix.
        embedding_dim (int): Dimensionality of the GloVe embeddings (e.g., 100).
    Returns:
        np.ndarray: Embedding matrix of shape (len(vocab), embedding_dim).
    """
    dir_path = os.path.dirname(glove_path) or '.'
    os.makedirs(dir_path, exist_ok=True)

    # Credit to ChatGPT for helping with the download and extraction logic
    # download and extract glove if not present
    if not os.path.isfile(glove_path):
        url = "http://nlp.stanford.edu/data/glove.6B.zip"
        zip_path = os.path.join(dir_path, "glove.6B.zip")
        print(f"Downloading GloVe embeddings from {url} to {zip_path}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as zp:
                for chunk in r.iter_content(chunk_size=8192):
                    zp.write(chunk)

        # extract
        target_file = f"glove.6B.{embedding_dim}d.txt"
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extract(target_file, dir_path)

        extracted_path = os.path.join(dir_path, target_file)
        if extracted_path != glove_path:
            os.replace(extracted_path, glove_path)
        os.remove(zip_path)

    # Read embeddings into a dictionary
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            vals = line.split()
            word = vals[0]
            vec = np.asarray(vals[1:], dtype='float32')
            embeddings[word] = vec

    # Build embedding matrix aligned to vocab
    matrix = np.zeros((len(vocab), embedding_dim), dtype='float32')
    for word, idx in vocab.items():
        if word in embeddings:
            matrix[idx] = embeddings[word]
    return matrix


def stratified_multilabel_split(X, y, splits=[0.6, 0.2, 0.2], random_state=42):
    """
    Stratified multi-label data splitter.

    Parameters:
    - X (array-like): Features or input text.
    - y (array-like): Multi-label targets.
    - splits (list): Fractions summing to 1.0. Supports [train, val], or [train, val, test].
    - random_state (int): Random seed for reproducibility.

    Returns:
    - List of (X_part, y_part) for each split.
    """
    assert np.isclose(sum(splits), 1.0), "Splits must sum to 1.0"
    assert len(splits) in [2, 3], "Only 2 or 3 splits are supported"
    X_train = y_train = X_val = y_val = X_test = y_test = None
    X = np.array(X)
    y = np.array(y)
    if len(splits) == 2:
        # 2-way split: train, val
        train_frac = splits[0]
        mskf = MultilabelStratifiedKFold(n_splits=int(1 / (1 - train_frac)),
                                         shuffle=True, random_state=random_state)

        for train_idx, val_idx in mskf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

        return [(X_train, y_train), (X_val, y_val)]

    elif len(splits) == 3:
        # 3-way split: train, val, test
        train_frac, val_frac, test_frac = splits

        # Step 1: test split
        mskf1 = MultilabelStratifiedKFold(n_splits=int(1 / test_frac),
                                          shuffle=True, random_state=random_state)

        for train_val_idx, test_idx in mskf1.split(X, y):
            X_train_val, X_test = X[train_val_idx], X[test_idx]
            y_train_val, y_test = y[train_val_idx], y[test_idx]
            break

        # Step 2: train/val split from remaining
        adjusted_train_frac = train_frac / (train_frac + val_frac)
        mskf2 = MultilabelStratifiedKFold(n_splits=int(1 / (1 - adjusted_train_frac)),
                                          shuffle=True, random_state=random_state)

        for train_idx, val_idx in mskf2.split(X_train_val, y_train_val):
            X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
            y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
            break

        return [(X_train, y_train), (X_val, y_val), (X_test, y_test)]
    else:
        return [(X_train, y_train), (X_val, y_val), (X_test, y_test)]


class ToxicDataset(Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]
