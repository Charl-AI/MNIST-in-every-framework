#!usr/bin/env bash

mkdir -p data
cd data

# download datasets
kaggle competitions download -p kaggle_mnist -c digit-recognizer
kaggle competitions download -p kaggle_tweets -c nlp-getting-started
kaggle competitions download -p kaggle_titanic -c titanic

# prepare datasets
unzip kaggle_mnist/digit-recognizer.zip -d kaggle_mnist
unzip kaggle_tweets/nlp-getting-started.zip -d kaggle_tweets
unzip kaggle_titanic/titanic.zip -d kaggle_titanic
