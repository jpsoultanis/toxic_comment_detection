class Config:
    EMBEDDING_DIM = 100
    HIDDEN_SIZE = 64
    MAX_LEN = 100
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-3
    EMBEDDING_PATH = f"dataset/word_embeddings/glove.6B/glove.6B.100d.txt"
    MODEL_PATH = f"bilstm.pth"
    DATA_PATH = f"dataset/.cache/kagglehub/competitions/jigsaw-toxic-comment-classification-challenge/train.csv.zip"
    TEST_FILE_PATH = "dataset/.cache/kagglehub/competitions/jigsaw-toxic-comment-classification-challenge/test.csv.zip"
    TEST_LABEL_PATH = "dataset/.cache/kagglehub/competitions/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip"
    LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    SAVE_TEST_CSV_PATH = "test_file.csv"
    SEED = 42