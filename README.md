# Yikes Detector

A final project for the course `Deep Learning CS7643 (OMSCS)` by
- Divyanshu Singhal
- Haimanot Yibeltal Tiruneh
- Joseph Soultanis
- Shekhar Koirala

### Documents
[Proposal Link](./docs/Project_Proposal_CS7643.pdf)

[Report Link](https://www.overleaf.com/project/6803e6c042c4ea4eeca73fea)


### Setting up dependencies locally
##### 1. Install [uv (link)](https://docs.astral.sh/uv/getting-started/installation/)

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

##### 2. Setting up the virtual Env
```
uv sync
```

##### 3. setup pre-commit
```
source .venv/bin/activate

pre-commit install
```

### 4. Downloading the dataset

Download kaggle.json from [https://www.kaggle.com/settings](https://www.kaggle.com/settings)
Under the API section, a `kaggle.json` file will be downloaded. Keep it in ~/.kaggle/kaggle.json

#### File descriptions

`train.csv` - the training set, contains comments with their binary labels

`test.csv` - the test set, you must predict the toxicity probabilities for these comments. To deter hand labeling, the test set contains some comments which are not included in scoring.

`sample_submission.csv` - a sample submission file in the correct format

`test_labels.csv` - labels for the test data; value of -1 indicates it was not used for scoring
