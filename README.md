# Advanced Machine Learning Application - Spring 2025 - AT1 - Kaggle Competition

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

End-to-end workflow to predict whether a player will be drafted using detailed player statistics, including box score metrics (points, assists, rebounds, blocks), advanced efficiency measures (offensive/defensive ratings, usage rate, shooting percentages), and contextual features (team, conference, year, height). The target variable is whether a player was drafted (binary outcome), with the dataset being highly imbalanced since only a small fraction of players make it to the draft.. The repo contains four experiment notebooks (Exp-0 … Exp-3), saved model artefacts under models/, and utilities in src/amla_at1.

-----

## Project Structure

```
amla_at1/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   ├── 36120-25SP-AT1-group12-25238736-experiment-0.ipynb
│   ├── 36120-25SP-AT1-group12-25238736-experiment-1.ipynb
│   ├── 36120-25SP-AT1-group12-25238736-experiment-2.ipynb
│   └── 36120-25SP-AT1-group12-25238736-experiment-3.ipynb
├── models/
│   ├── exp4_catboost_20250829_044226/           # CatBoost .cbm + feature_importance + schema
│   └── exp2_best_20250829-044334/       # sklearn pipeline + metadata
├── src/
│   └── amla_at1/    
│       ├── data/
│       ├── features/
│       └── models/
├── test/
│   ├── data/    
│   ├── features/
│   └── models/                      
├── pyproject.toml
├── requirements.txt
├── dist/
├── poetry.lock
└── README.md

```

--------

## Requirements

- **`Python`**: 3.11.4
- **Key libraries**: 
    - `pandas` = 2.2.2
    - `jupyterlab` = 4.2.3
    - `scikit-learn` = 1.5.1
    - `joblib` = 1.4.2
    - `xgboost` = 2.1.0
    - `hyperopt` = 0.2.7
    - `lightgbm` = 4.4.0
    - `lime` = 0.2.0.1
    - `wandb` = 0.17.4
    - `numpy` = 1.26.4
    - `scipy` = 1.11.4
    - `catboost` = 1.2.5
    - `pytest` = 8.2.2

## Setup

> Use either **Poetry** (recommended) or **pip/venv**.

### 1) Clone

```bash
git clone https://github.com/naynajn/amla_at1.git
cd amla_at1
```
### 2) Set Python 3.11.4 with pyenv

```bash
pyenv install 3.11.4
pyenv local 3.11.4
python -V
```
### 3A) Install with Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
# or: pipx install poetry
```

### 3B) Install dependencies & create venv
```bash
poetry install
poetry shell
```

## Launch JupyterLab
```bash
poetry run jupyter lab
```
----
## Running the Experiments
Open the notebooks in notebooks/ and run top-to-bottom:
- Exp-0: Baseline data understanding + baseline model
- Exp-1: Feature selection (Mutual Information + Logistic(OHE))
- Exp-2: Tree-based selection (LightGBM gain + permutation) and improvements
- Exp-3: Gradient boosting ensemble algorithm (e.g., CatBoost), L1 selector consolidation, calibration & diagnostics
-----

## Example: Using the Helper Package

```bash
from pathlib import Path
import pandas as pd

from amla_at1.models.performance import metrics_from_proba
from amla_at1.features.dates import safe_to_ordinal
from amla_at1.data.sets import train_val_test_split_like

DATA_DIR = Path("data")
df = pd.read_csv(DATA_DIR / "train.csv")

# ... preprocess, fit your model, get predicted probabilities 'p'
# metrics = metrics_from_proba(p, df['drafted'].values)

```
#### The project uses the src/ layout so the package is available when the repo is installed with Poetry or when PYTHONPATH includes src.
-----

## Saving Model Artefacts
All best models should save into models/<run_name_timestamp>/:
- Model file: `.joblib` (`sklearn`) or `.cbm` (`CatBoost`)
- `feature_importance.csv`: ranked importance where available
- `schema.json`: expected input features (and any preprocessing notes)
- `metadata.json`: metrics, split info, random seeds, code/hash, library versions

-----
## Generating a Kaggle Submission
#### Kaggle expects: player_id and drafted with drafted as probabilities.

```bash
import numpy as np
import pandas as pd
from pathlib import Path

ID_COL = "player_id"
DATA_DIR = Path("data")

test = pd.read_csv(DATA_DIR / "test.csv")
X_submit = test.drop(columns=[ID_COL], errors="ignore").copy()

probs = pipe.predict_proba(X_submit)[:, 1]
probs = np.clip(probs, 1e-9, 1 - 1e-9)

sub = pd.DataFrame({ID_COL: test[ID_COL].astype(str), "drafted": probs})
out_path = Path("notebooks") / "submission_expX_YYYYMMDD.csv"
sub.to_csv(out_path, index=False)
print(sub.head())
```

-----

## Running Tests

```bash
poetry run pytest -q
```
-----
## Acknowledgements
Project layout inspired by the Cookiecutter Data Science style.

-----