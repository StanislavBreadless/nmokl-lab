import tensorflow as tf;
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import os

vocab_size = 50000
embedding_dim = 10
max_length = 1000
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

TRAINING_SIZE = 9000

import re
# as per recommendation from @freylis, compile once only
CLEANR = re.compile('<.*?>') 

def cleanhtml(raw_html):
  cleantext = re.sub(CLEANR, ' ', raw_html)
  return cleantext


def process(txts):
  for i in range(len(txts)):
    txts[i] = cleanhtml(txts[i])
  return txts

def get_data(type):
    # return []
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



print('Tokenization')

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
# training_data = process(['<br /><br />', 'I do not care about the dogs!!!'])
training_data = process(training_data)
# print(training_data)
# quit()
tokenizer.fit_on_texts(training_data)



word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_data)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# print(training_padded)
# quit()

testing_sequences = tokenizer.texts_to_sequences(testing_data)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)



import numpy as np
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

print('Tokenization complete. Compiling model...')

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.SpatialDropout1D(0.3),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(12, activation='relu', kernel_regularizer='l2'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
#     tf.keras.layers.SpatialDropout1D(0.15),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
#     tf.keras.layers.Dense(12, activation='relu', kernel_regularizer='l2'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())


num_epochs = 30
print(f'Training model. Number of epochs = {num_epochs}')

history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)

# import numpy as np
# training_padded = np.array(training_padded)
# training_labels = np.array(training_labels)
# testing_padded = np.array(testing_padded)
# testing_labels = np.array(testing_labels)
