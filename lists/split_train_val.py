import os
import sys
import pandas as pd
from sklearn import model_selection

sys.path.append("./libs")

if "utils" not in sys.modules:
    from utils.random_seed import SEED


ROOT = "./lists/"
SAVE_DIR = "./lists/splits/"
DATA_PATH = os.path.join(ROOT, "train.csv")

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    train_df, val_df = model_selection.train_test_split(
        df, test_size=0.1, random_state=SEED
    )
    train_df.to_csv(f"{SAVE_DIR}train.csv", index=False)
    val_df.to_csv(f"{SAVE_DIR}val.csv", index=False)
