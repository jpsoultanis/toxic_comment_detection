# python based modules
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# custom modules

from config import Config
from data_cleaning import preprocess_and_tokenize, load_glove_embeddings, ToxicDataset, stratified_multilabel_split
from train import set_seed, CBFocalLoss, train_one_epoch, evaluate
from model import ToxicCommentClassifier
from inference import run_inference, prepare_test_data

def run_and_infer():
    cfg = Config()
    set_seed(Config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and process data
    df = pd.read_csv(cfg.DATA_PATH)
    df, vocab = preprocess_and_tokenize(df, cfg.MAX_LEN)
    embedding_matrix = load_glove_embeddings(cfg.EMBEDDING_PATH, vocab, cfg.EMBEDDING_DIM)
    samples_per_cls = torch.tensor([df[col].sum() for col in cfg.LABELS], dtype=torch.float32)

    X =  df['input_ids'].tolist()
    y = df[cfg.LABELS].values.tolist()

    splits  = stratified_multilabel_split(X, y, splits=[0.6, 0.2, 0.2], random_state=42)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = splits

    train_ds = ToxicDataset(X_train, y_train)
    val_ds = ToxicDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE)

    # Model setup
    model = ToxicCommentClassifier(len(vocab), cfg.EMBEDDING_DIM, embedding_matrix,
                                   hidden_size=cfg.HIDDEN_SIZE,
                                   output_dim=len(cfg.LABELS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    # Scheduler (track validation F1 to reduce LR when F1 plateaus)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    criterion = CBFocalLoss(samples_per_cls=samples_per_cls, beta=0.9999, gamma=2.0)

    # Training loop
    best_f1 = 0.0
    for epoch in range(20):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_f1 = evaluate(model, val_loader, device)
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), cfg.MODEL_PATH)
        # 🔥 Step the scheduler
        scheduler.step(val_f1)
        print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")

    # Load test data
    # USE THIS IF YOU ARE DOING TESTING ON TEST FILE DATA
    # test_df = pd.read_csv(cfg.TEST_FILE_PATH,  dtype={"id": str})
    # test_labels_df = pd.read_csv(cfg.TEST_LABEL_PATH,  dtype={"id": str})  # Optional

    # USE THIS IF YOU ARE DOING 60-20-20
    test_ds = ToxicDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE)

    # test_ds = ToxicDataset(test_df['input_ids'].tolist(), dummy_labels)
    # test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE)

    # Load trained model
    # checkpoint = torch.load('custom_model_bilstm.pth', map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Run inference
    preds = run_inference(model, test_loader, device)
    print(classification_report(y_test, preds, target_names=cfg.LABELS, zero_division=0))

    # DO THIS IF YOU WANT TO PLAY WITH TEST DATA
    # Save predictions
    # output_df = test_df.copy()
    # output_df[cfg.LABELS] = preds
    # output_df.to_csv(cfg.SAVE_TEST_CSV_PATH, index=False)
    # true_labels = test_labels_df[cfg.LABELS].values
    # mask = (true_labels < 0).any(axis=1)
    # true_labels_copy = true_labels[~mask]
    # pred_labels = preds[~mask]
    #
    # print(classification_report(true_labels_copy, pred_labels, target_names=cfg.LABELS, zero_division=0))


if __name__ == '__main__':
    run_and_infer()