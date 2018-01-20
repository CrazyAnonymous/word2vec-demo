import logging

from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Activation
from sklearn.cross_validation import train_test_split

from utils.text_utils import text_to_words, words_to_matrix

logging.getLogger(__name__).setLevel(logging.INFO)

# 加载简爱小说并分词
words = text_to_words('data/Jane_eyer.txt')

max_words = len(words)

# 划分训练集和测试集，此时都是list列表
X_train_words, X_test_words, y_train_words, y_test_words = train_test_split(words, words, test_size=0.2)

logging.info('training and test data loaded.')

X_train = words_to_matrix(X_train_words)
y_train = words_to_matrix(y_train_words)
X_test = words_to_matrix(X_test_words)
y_test = words_to_matrix(y_test_words)


embedding_layer = Embedding(max_words, 100, weights=[X_train], input_length=max_words, trainable=False, dropout=0.2)

model = Sequential()
model.add(embedding_layer)
model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))
# model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

logging.info('training start')

model.fit(X_train, y_train, batch_size=32, nb_epoch=5, validation_data=(X_test, y_test))

score, acc = model.evaluate(X_test, y_test, batch_size=100)

logging.info('score: %s' % score)
logging.info('test accuracy: %s' % acc)
