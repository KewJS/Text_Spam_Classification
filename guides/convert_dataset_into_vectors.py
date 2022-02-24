import pathlib
import pickle
import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

USE_PROJECT_ROOT = True
BASE_DIR = pathlib.Path().resolve()
if USE_PROJECT_ROOT:
    BASE_DIR = BASE_DIR.parent
    
DATASET_DIR = BASE_DIR / "datasets"
EXPORT_DIR = DATASET_DIR / "exports"
DATASET_CSV_PATH = EXPORT_DIR / 'spam-dataset.csv'
TRAINING_DATA_PATH = EXPORT_DIR / 'spam-training-data.pkl'
print("BASE_DIR is", BASE_DIR)

RUN_DATASET_PREPARE = False
if RUN_DATASET_PREPARE:
    # if active, this will download and prepare the dataset.
    SOURCE_NB = pathlib.Path('01-Prepare_AI_Spam_Classifier_Dataset.ipynb')
    if SOURCE_NB.exists():
        %run './{SOURCE_NB}'
    else:
        print("01-Prepare_AI_Spam_Classifier_Dataset.ipynb does not exist.")
        
if not DATASET_CSV_PATH.exists():
    raise Exception(f"You must download or create the spam-dataset.csv \n{DATASET_CSV_PATH} not found.")

df = pd.read_csv(str(DATASET_CSV_PATH))

texts = df['text'].tolist()
labels = df['label'].tolist()

labels_legend = {'ham': 0, 'spam': 1}
labels_legend_inverted = {f"{v}":k for k,v in labels_legend.items()}
labels_as_int =  [labels_legend[str(x)] for x in labels]

random_idx = random.randint(0, len(texts))
print('Random Index', random_idx)

assert texts[random_idx] == df.iloc[random_idx].text
assert labels[random_idx] == df.iloc[random_idx].label
assert labels_legend_inverted[str(labels_as_int[random_idx])] == labels[random_idx]

MAX_NUM_WORDS = 280
MAX_SEQUENCE_LENGTH = 300

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

assert len(sequences) == len(texts) == len(labels_as_int)

X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
y = to_categorical(np.asarray(labels_as_int))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
training_data = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'max_words': MAX_NUM_WORDS,
    'max_sequence': MAX_SEQUENCE_LENGTH,
    'legend': labels_legend,
    'labels_legend_inverted': labels_legend_inverted,
    "tokenizer": tokenizer,
}

with open(TRAINING_DATA_PATH, 'wb') as f:
    pickle.dump(training_data, f)