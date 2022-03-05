import os
import pandas as pd


def load_and_prepare(data_dir="data/kaggle_titanic"):
    train_val_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    # Using same preprocessing steps as:
    # https://www.kaggle.com/shubhendumishra/titanic-mlp-model-using-pytorch
    data = [train_val_df, test_df]
    for dataset in data:
        dataset["relatives"] = dataset["SibSp"] + dataset["Parch"]

    train_val_df.drop(
        ["SibSp", "Parch", "Ticket", "Name", "Cabin"], axis=1, inplace=True
    )
    test_df.drop(["SibSp", "Parch", "Ticket", "Name", "Cabin"], axis=1, inplace=True)
    train_val_df["Sex"] = train_val_df["Sex"].map(dict(zip(["male", "female"], [0, 1])))
    test_df["Sex"] = test_df["Sex"].map(dict(zip(["male", "female"], [0, 1])))
    age_mean = train_val_df["Age"].mean()
    train_val_df["Age"] = train_val_df["Age"].fillna(age_mean)
    train_val_df.dropna(inplace=True)
    train_val_df["Embarked"] = train_val_df["Embarked"].map(
        dict(zip(["S", "C", "Q"], [0, 1, 2]))
    )
    test_df["Embarked"] = test_df["Embarked"].map(dict(zip(["S", "C", "Q"], [0, 1, 2])))

    min_age = train_val_df["Age"].min()
    max_age = train_val_df["Age"].max()

    train_val_df["Age"] = train_val_df["Age"] / max_age

    test_min_age = test_df["Age"].min()
    test_max_age = test_df["Age"].max()

    test_df["Age"] = test_df["Age"] / test_max_age

    min_age = train_val_df["Fare"].min()
    max_age = train_val_df["Fare"].max()

    train_val_df["Fare"] = train_val_df["Fare"] / max_age

    test_min_age = test_df["Fare"].min()
    test_max_age = test_df["Fare"].max()

    test_df["Fare"] = test_df["Fare"] / test_max_age

    train_val_df["relatives"] = train_val_df["relatives"].apply(
        lambda x: 3 if x >= 3 else x
    )


def get_data_batches():
    """Returns a iterables of NumPy arrays (train_batches, val_batches, test_batches)"""
    pass
