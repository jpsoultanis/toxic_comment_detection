import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

from typing import Optional
import os


class JigsawTrainDataset(Dataset):
    def __init__(self, dataframe: Optional[pd.DataFrame] = None, csv_file: Optional[os.PathLike] = None, tokenizer_name='bert-base-uncased', max_len=128):

        if dataframe is not None:
            self.data = dataframe
        elif csv_file:
            self.data = pd.read_csv(csv_file)
        else:
            raise ValueError("You must provide either a DataFrame or a CSV file path")

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.texts = self.data['comment_text'].fillna("").tolist()
        self.labels = self.data[[
            'toxic', 'severe_toxic', 'obscene',
            'threat', 'insult', 'identity_hate'
        ]].values.astype('float32')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float)

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': label
        }

class JigsawTestDataset(Dataset):
    def __init__(self, test_csv: os.PathLike, tokenizer_name='bert-base-uncased', max_len=128):
        self.data = pd.read_csv(test_csv)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.texts = self.data['comment_text'].fillna("").tolist()
        self.ids = self.data['id'].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        sample_id = self.ids[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'id': sample_id,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


def load_train_val_data(csv_path: os.PathLike, tokenizer_name='bert-base-uncased', max_len=128, val_size=0.1, batch_size=32):
    # Load full training CSV
    df = pd.read_csv(csv_path)

    # Train/Val Split
    train_df, val_df = train_test_split(df, test_size=val_size, random_state=42)

    # Create datasets
    train_dataset = JigsawTrainDataset(dataframe=train_df, tokenizer_name=tokenizer_name, max_len=max_len)
    val_dataset = JigsawTrainDataset(dataframe=val_df, tokenizer_name=tokenizer_name, max_len=max_len)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# if __name__ == "__main__":
#     train_csv = ".cache/kagglehub/competitions/jigsaw-toxic-comment-classification-challenge/train.csv.zip"
#     train_loader, val_loader = load_train_val_data(train_csv)
#
#     for batch in train_loader:
#         print("Train batch:", batch['input_ids'].shape)
#         break
#
#     for batch in val_loader:
#         print("Validation batch:", batch['input_ids'].shape)
#         break