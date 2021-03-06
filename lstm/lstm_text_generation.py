'''
Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function

import jieba
from keras_demo.models import Sequential
from keras_demo.layers import Dense, Activation
from keras_demo.layers import LSTM
from keras_demo.optimizers import RMSprop
from keras_demo.utils.data_utils import get_file
import numpy as np
import random
import sys
import io


# path = get_file('C:/Workbench/Python/word2vec-demo/lstm/data/nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
path = get_file('C:/Workbench/Python/word2vec-demo/lstm/data/Jane_eyer.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')

text = io.open(path, encoding='utf-8').read().lower()
cuted_text = jieba.cut(text)
text_arr = []

for t in cuted_text:
    text_arr.append(t)

print('corpus length:', len(text_arr))

chars = sorted(list(set(text_arr)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text_arr) - maxlen, step):
    sentences.append(text_arr[i: i + maxlen])
    next_chars.append(text_arr[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(1, 2):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x, y,
              batch_size=128,
              epochs=1)

    start_index = random.randint(0, len(text_arr) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = []
        sentence = text_arr[start_index: start_index + maxlen]
        generated.append(sentence)
        print('----- Generating with seed: "' + str(sentence) + '"')
        sys.stdout.write(str(generated))

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            index = 0
            for t, char in enumerate(sentence):
                if index < 40:
                    x_pred[0, t, char_indices[char]] = 1.
                    index = index + 1

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated.append(next_char)
            # sentence = sentence[1:] + next_char
            sentence.append(next_char)

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()