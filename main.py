import tensorflow as tf;
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import os

# The maximum amount of n-grams allowed
vocab_size = 50000
# The dimension of the embedding layer
embedding_dim = 10
# Maximum length of texts (the rest are padded/truncated) to this length
max_length = 1000
# Truncation will delete the last characters
trunc_type='post'
# Padding will append zeroes to the end
padding_type='post'

# The size of training/testing datasets
TRAINING_SIZE = 12500

import re
# as per recommendation from @freylis, compile once only
CLEANR = re.compile('<.*?>') 

# Remove html tags from a text
def cleanhtml(raw_html):
  cleantext = re.sub(CLEANR, ' ', raw_html)
  return cleantext

# Removes html tags from an array of texts 
def process(txts):
  for i in range(len(txts)):
    txts[i] = cleanhtml(txts[i])
  return txts

# Reads texts based on type
def get_data(type):
    path = '.\\aclImdb\\' + type

    directory = os.fsencode(path)
    
    texts = []

    dd = os.listdir(directory)

    for file in dd[:TRAINING_SIZE//2]:
        filename = os.fsdecode(file)
        f = open(path + "\\" + filename, "r", encoding="utf8")
        texts.append(f.read())

    return texts


print('Reading training positive data...')
train_pos = get_data('train\\pos')
print('Reading training negative data...')
train_neg = get_data('train\\neg')

print('Reading testing positive data...')
testing_pos = get_data('test\\pos')
print('Reading testing negative data...')
testing_neg = get_data('test\\neg')

print('All data loaded to memory! Starting preprocessing...')

print('Assigning labels')
training_data = train_pos + train_neg
training_labels = []

for i in range(0, len(train_pos)):
  training_labels.append(1)

for i in range(0, len(train_neg)):
  training_labels.append(0)

testing_data = testing_pos + testing_neg
testing_labels = []

for i in range(0, len(testing_pos)):
  testing_labels.append(1)

for i in range(0, len(testing_neg)):
  testing_labels.append(0)

print(f'Labels assigned. Total training size = {len(training_labels)}. Positive labels = {len(train_pos)}. Negative labels = {len(train_neg)}')
print(f'Labels assigned. Total testing size = {len(testing_data)}. Positive labels = {len(testing_pos)}. Negative labels = {len(testing_neg)}')

print('Preprocessing')

# ngram_range means that we will allow ngrams of length 1,2,3
# binary is set to true to not account the same ngram twice. In our particular model 
# this is a better approach.
veczr =  CountVectorizer(ngram_range=(1,3), binary=True, 
                          token_pattern=r'\w+',
                          max_features=vocab_size)

training_data = process(training_data)

# Converts DTM to a sequences padded to maxlen
def dtm_to_seq(dtm, maxlen):
    x = []
    for _, row in enumerate(dtm):
        seq = []
        # Indeces numerated starting from 1.
        indices = (row.indices + 1).astype(np.int64)
        # Data inside the row
        data = (row.data).astype(np.int64)

        # Iterative over a tuple and filling in the sequence
        count_dict = dict(zip(indices, data))
        for k,v in count_dict.items():
            seq.extend([k]*v)
        num_words = len(seq)
        # Pad up to maxlen with 0
        if num_words < maxlen: 
            seq = np.pad(seq, (maxlen - num_words, 0),    
                         mode='constant')
        # Truncate down to maxlen
        else:                  
            seq = seq[-maxlen:]
        x.append(seq)
    return np.array(x)

# Converting training/testing data to DTMs
dtm_train = veczr.fit_transform(training_data)
dtm_test = veczr.transform(testing_data)

# Converting DTMs to sequence of words 
seq_train = dtm_to_seq(dtm_train, max_length)
seq_test = dtm_to_seq(dtm_test, max_length)

training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)

print('Preprocessing complete. Compiling model...')

model = tf.keras.Sequential([
    # +1 is needed, since values in range [0, vocab_size] are possible, where
    # [1, vocab_size] correspond to the words, while "0" means no item
    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(12, activation='relu', kernel_regularizer='l2'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())

num_epochs = 30
print(f'Training model. Number of epochs = {num_epochs}')

history = model.fit(seq_train, training_labels, epochs=num_epochs, validation_data=(seq_test, testing_labels), verbose=2)
