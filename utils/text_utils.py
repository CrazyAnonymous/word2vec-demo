import jieba
from gensim import corpora
from gensim.matutils import corpus2dense
import numpy as np


def file_to_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def file_to_words(path):
    """
    将文本文件转换为sequences
    :param path:
    :return:
    """
    sentences = []
    with open(path, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            words = jieba.lcut(line, cut_all=True, HMM=True)
            sentences.append(words)
            line = f.readline()

    return list(sentences)


def text_to_words(text):
    """
    将文本转换为sequences
    :param text:
    :return:
    """
    sentences = []
    words = jieba.lcut(text, cut_all=True, HMM=True)
    sentences.append(words)

    return list(sentences)


def sentences_to_matrix(sentences):
    """
    将二维词组转换为Matrix
    Example: [["从前", "有", "个"], ["山", "里面"]]
    :param sentences:
    :return:
    """
    # 将 sentences 去重并转换为dict
    dictionary = corpora.Dictionary(sentences)
    word_shape = [dictionary.doc2bow(sentence) for sentence in sentences]
    matrix = corpus2dense(word_shape, len(dictionary))
    return matrix


def words_to_matrix(words):
    """
    将一维词组转换为 Matrix
    :param words:
    :return:
    """
    dictionary = corpora.Dictionary(words)
    word_shape = [dictionary.doc2bow(word) for word in words]
    matrix = corpus2dense(word_shape, len(dictionary))
    return matrix


def word_to_matrix(vocabulary, word):
    """
    将单词转换为 matrix
    :param vocabulary:
    :param word:
    :return:
    """
    id = vocabulary.get(word)
    if id is None:
        id = 0

    return id


def text_to_matrix(vocabulary, text):
    """
    将文章转换为matrix array
    :param vocabulary:
    :param text:
    :return:
    """
    words = jieba.lcut(text, cut_all=True, HMM=True)
    arr = []
    for word in words:
        arr.append(word_to_matrix(vocabulary, word))
    # x = list(map(word_to_matrix, words))
    return np.array(arr)


if __name__ == '__main__':
    words = file_to_words('C:/Workbench/Python/word2vec-demo/lstm/data/neg.txt')
    words_matrix = sentences_to_matrix(words)
    print(words)
    print(words_matrix)


