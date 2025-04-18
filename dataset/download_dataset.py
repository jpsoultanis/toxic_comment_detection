import os
from pathlib import Path
import kagglehub

# IMPORTANT ::::::  JOIN the competition
# download kaggle.json from  https://www.kaggle.com/settings and store it in ~/.kaggle/kaggle.json
# or input the data via login function by uncommenting.

# kagglehub.login()

DATASET_DIR = Path(__file__).resolve().parent

(DATASET_DIR / ".cache" / "kagglehub").mkdir(parents=True, exist_ok=True) # for downloading the dataset
os.environ["KAGGLEHUB_CACHE"] = str(DATASET_DIR / ".cache" / "kagglehub", )
download_path = os.environ["KAGGLEHUB_CACHE"]



def download_toxic_bias_from_kaggle():
    competition_name = "jigsaw-unintended-bias-in-toxicity-classification"
    kagglehub.competition_download(handle=competition_name, path= "train.csv", force_download=False)
    kagglehub.competition_download(handle=competition_name, path= "test.csv", force_download=False)
    kagglehub.competition_download(handle=competition_name, path= "all_data.csv", force_download=False)
    kagglehub.competition_download(handle=competition_name, path= "toxicity_individual_annotations.csv", force_download=False)
    kagglehub.competition_download(handle=competition_name, path= "identity_individual_annotations.csv", force_download=False)
    kagglehub.competition_download(handle=competition_name, path= "test_public_expanded.csv", force_download=False)
    kagglehub.competition_download(handle=competition_name, path= "test_private_expanded.csv", force_download=False)
    # verify the download
    print(f"files in the kaggle download folder: {os.listdir(os.path.join(download_path, 'competitions', competition_name))}")

def download_toxic_comment_from_kaggle():
    competition_name = "jigsaw-toxic-comment-classification-challenge"
    kagglehub.competition_download(handle=competition_name, path= "train.csv.zip", force_download=False)
    kagglehub.competition_download(handle=competition_name, path= "test.csv.zip", force_download=False)
    kagglehub.competition_download(handle=competition_name, path= "test_labels.csv.zip", force_download=False)
    kagglehub.competition_download(handle=competition_name, path= "sample_submission.csv.zip", force_download=False)
    # verify the download
    print(f"files in the kaggle download folder: {os.listdir(os.path.join(download_path, 'competitions', competition_name))}")


if __name__ == "__main__":
    download_toxic_comment_from_kaggle()
