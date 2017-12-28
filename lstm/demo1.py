import jieba  # 导入结巴分词
import numpy as np  # 导入Numpy
import pandas as pd  # 导入Pandas
from keras.layers import LSTM, Embedding
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils import np_utils

# from __future__ import absolute_import #导入3.x的特征函数
# from __future__ import print_function

neg = pd.read_excel('data/neg.xls', header=None, index=None)
pos = pd.read_excel('data/pos.xls', header=None, index=None)  # 读取训练语料完毕
pos['mark'] = 1
neg['mark'] = 0  # 给训练语料贴上标签
pn = pd.concat([pos, neg], ignore_index=True)  # 合并语料
neglen = len(neg)
poslen = len(pos)  # 计算语料数目

cw = lambda x: list(jieba.cut(x))  # 定义分词函数
pn['words'] = pn[0].apply(cw)

comment = pd.read_excel('data/sum.xls')  # 读入评论内容
# comment = pd.read_csv('a.csv', encoding='utf-8')
comment = comment[comment['rateContent'].notnull()]  # 仅读取非空评论
comment['words'] = comment['rateContent'].apply(cw)  # 评论分词

d2v_train = pd.concat([pn['words'], comment['words']], ignore_index=True)

# 将所有词语整合在一起
w = []

for i in d2v_train:
    w.extend(i)

# 统计词出现的次数
dict = pd.DataFrame(pd.Series(w).value_counts())

del w, d2v_train

dict['id'] = list(range(1, len(dict) + 1))
get_sent = lambda x: list(dict['id'][x])

pn['sent'] = pn['words'].apply(get_sent)

maxlen = 50
print('Pad sequences (samples x time)')
pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))

# 训练集
x = np.array(list(pn['sent']))[::2]
y = np.array(list(pn['mark']))[::2]

# 测试集
xt = np.array(list(pn['sent']))[::2]
yt = np.array(list(pn['mark']))[1::2]

# 全集
xa = np.array(list(pn['sent']))
ya = np.array(list(pn['mark']))

print('Build model...')

model = Sequential()
# try using a GRU instead, for fun
model.add(Embedding(len(dict) + 1, 256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')
# 训练时间为若干个小时
model.fit(x, y, batch_size=16, nb_epoch=10)

classes = model.predict_classes(xt)
acc = np_utils.accuracy(classes, yt)

print('Test accuracy:', acc)
