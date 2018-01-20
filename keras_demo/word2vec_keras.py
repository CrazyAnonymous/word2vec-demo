from gensim.models import Word2Vec
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense
from utils.text_utils import text_to_matrix, file_to_text


# 导入sogou model
word2vec_model = Word2Vec.load('E:/Word2Vec/sougou/corpus.model')

# 获取权重
weights = word2vec_model.wv.syn0

# 获得词库
vocabulary = dict([(k, v.index) for k, v in word2vec_model.wv.vocab.items()])

# x = '我们刚到县城的第一年，发生了一次月食。那天夜晚，凉风习习，我家与汤姆家，大人小孩们，都来到楼房平顶上，坐在凉席上，聊天吹风，等待暑热消散。我和汤姆都很兴奋，仰躺在凉爽的竹席上，瞪大了眼睛，等待神奇的“天狗吃月亮”。'
# y = '流浪汉听到声音，抬起头，看着我们傻笑，我吓得缩回脑袋。很多孩子，都在放肆地笑，汤姆却没笑，她下了去跑回家，拿出几袋雪饼。看着汤姆向流浪汉走去，我紧张得不敢说话，大孩子们也不敢笑了。汤姆小跑了几步，追上流浪汉，那张黢黑而肮脏的脸转了过来。汤姆将雪饼递出去，那人看起来有些害怕，接之前，还犹豫了一下。'

x = y = file_to_text('../lstm/data/Jane_eyer.txt')

# x_train = to_ids(x)
# y_train = to_ids(x)
# x_test = to_ids(y)
# y_test = to_ids(y)
# 训练数据
x_train = text_to_matrix(vocabulary, x)
x_test = text_to_matrix(vocabulary, y)

# 标签数据
y_train = text_to_matrix(vocabulary, x)
y_test = text_to_matrix(vocabulary, y)
# x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.2)

# 做为 Embedding 层
embedding_layer = Embedding(input_dim=weights.shape[0],
                            output_dim=weights.shape[1],
                            weights=[weights], trainable=False)

# keras_demo
keras_model = Sequential()
keras_model.add(embedding_layer)
keras_model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
keras_model.add(Dense(1, activation='sigmoid'))


# Cross Entropy (交叉熵)， 和 Adam Optimizer
keras_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 训练模型，以每批次32样本迭代数据
history = keras_model.fit(x_train, x_train, validation_data=(x_test, x_test), epochs=20, batch_size=10, verbose=1)
# 计算输入模型在某些输入数据上的偏差
loss, accuracy = keras_model.evaluate(x_test, y_test)
print("loss: %s, accuracy: %s" % (loss, accuracy))
