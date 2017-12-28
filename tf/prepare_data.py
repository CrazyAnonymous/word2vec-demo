
def read_data():
    stop_word = []
    with open('E:/Word2Vec/sougou/corpus_seg.txt', 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            stop_word.append(line[:-1])
            line = f.readline()

    stop_word = set(stop_word)