import os
import pathlib
import pickle
import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

BASE_DIR = pathlib.Path().resolve().parent
DATASET_DIR = BASE_DIR / "data_local"
EXPORT_DIR = DATASET_DIR/"exports"
SPAM_DATASET_PATH = EXPORT_DIR/"spam-dataset.csv"

METADATA_EXPORT_PATH = EXPORT_DIR/"spam-metadata.pkl"
TOKENIZER_EXPORT_PATH = EXPORT_DIR/"spam-tokenizer.json"

MAX_NUM_WORDS = 280
MAX_SEQ_LENGTH = 300

model_df = pd.read_csv(SPAM_DATASET_PATH)
labels = model_df["label"].tolist()
texts = model_df["text"].tolist()

label_legend = {"ham": 0, "spam": 1}
label_legend_inverted = {f"{v}": k for k, v in label_legend.items()}
labels_as_int = [label_legend[x] for x in labels]
label_legend_inverted[str(labels_as_int[120])]

random_idx = random.randint(0, len(labels))

assert texts[random_idx] == model_df.iloc[random_idx]["text"]
assert labels[random_idx] == model_df.iloc[random_idx]["label"]
assert label_legend_inverted[str(labels_as_int[random_idx])] == model_df.iloc[random_idx].label

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

x = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH)
labels_as_int_array = np.asarray(labels_as_int)
y = to_categorical(labels_as_int_array)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
training_data = {
    "x_train": x_train,
    "x_test": x_test,
    "y_train": y_train,
    "y_test": y_test,
    "max_words": MAX_NUM_WORDS, 
    "max_seq_length": MAX_SEQ_LENGTH,
    "legend": label_legend,
    "label_legend_invereted": label_legend_inverted,
}

tokenizer_json = tokenizer.to_json()
TOKENIZER_EXPORT_PATH.write_text(tokenizer_json)

with open(METADATA_EXPORT_PATH, "wb") as f:
    pickle.dump(training_data, f)