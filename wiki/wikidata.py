import jieba

from opencc import OpenCC


class WikiData:
    def __init__(self, wiki_file):
        self.wiki_file = wiki_file

    def __iter__(self):
        openCC = OpenCC('t2s')
        for line in self.wiki_file:
            line = openCC.convert(line)
            s = jieba.cut(line)

            # 使用filter 过滤空格和换行，可以考虑同时过滤掉英文单词
            s = filter(lambda x: x != '' and x != '\n', s)
            yield list(s)

