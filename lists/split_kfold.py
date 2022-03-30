import os
import sys
import pandas as pd
from sklearn import model_selection

sys.path.append("./libs")

if "utils" not in sys.modules:
    from utils.random_seed import SEED

# ROOT = "./data/"
ROOT = "./lists/"
SAVE_DIR_TRAIN = "./lists/folds/train/"
SAVE_DIR_VAL = "./lists/folds/val/"
DATA_PATH = os.path.join(ROOT, "train_modified.csv")
NUM_FOLDS = 5

if __name__ == "__main__":
    try:
        os.makedirs(SAVE_DIR_TRAIN)
        os.makedirs(SAVE_DIR_VAL)
    except:
        pass
    df = pd.read_csv(DATA_PATH)
    skf = model_selection.StratifiedKFold(
        n_splits=NUM_FOLDS, random_state=SEED, shuffle=True
    )

    for idx, fold in enumerate(skf.split(df.image.values, df.individual_id.values)):
        train_idxs, val_idxs = fold
        train_df = {
            "image": df.image.values[train_idxs],
            "individual_id": df.individual_id.values[train_idxs]
        }
        val_df = {
            "image": df.image.values[val_idxs],
            "individual_id": df.individual_id.values[val_idxs]
        }
        train_df, val_df = (
            pd.DataFrame.from_dict(train_df),
            pd.DataFrame.from_dict(val_df)
        )
        train_df.to_csv(f"{SAVE_DIR_TRAIN}/train_{idx}.csv", index=False)
        val_df.to_csv(f"{SAVE_DIR_VAL}/val_{idx}.csv", index=False)
