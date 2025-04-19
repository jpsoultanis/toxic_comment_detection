import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertForSequenceClassification
from sklearn.metrics import f1_score
from tqdm import tqdm
import pandas as pd
from dataset.load_data import JigsawTrainDataset
from sklearn.model_selection import train_test_split


def load_dataloaders(csv_path, tokenizer_name='bert-base-uncased', max_len=128, val_size=0.1, batch_size=32):
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=val_size, random_state=42)

    train_dataset = JigsawTrainDataset(dataframe=train_df, tokenizer_name=tokenizer_name, max_len=max_len)
    val_dataset = JigsawTrainDataset(dataframe=val_df, tokenizer_name=tokenizer_name, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / len(val_loader)
    preds = torch.cat(all_preds).numpy()
    true = torch.cat(all_labels).numpy()

    preds_binary = (preds > 0.5).astype(int)
    f1 = f1_score(true, preds_binary, average='macro')

    return avg_loss, f1

def main():
    # Settings
    train_csv = "dataset/.cache/kagglehub/competitions/jigsaw-toxic-comment-classification-challenge/train.csv.zip"
    num_labels = 6
    epochs = 3
    batch_size = 16
    learning_rate = 2e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader = load_dataloaders(train_csv, batch_size=batch_size)

    # Load model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels,
        problem_type="multi_label_classification"
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    best_f1 = 0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss = train(model, train_loader, optimizer, device)
        val_loss, val_f1 = evaluate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

        # Save model if F1 improves
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "best_model.pt")
            print("🔥 Best model saved!")

if __name__ == "__main__":
    main()
