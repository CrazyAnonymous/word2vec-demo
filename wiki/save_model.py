import codecs
import multiprocessing

from gensim.models import Word2Vec

from wikidata import WikiData

if __name__ == '__main__':
    wiki_data = WikiData(codecs.open('./zh_tw_zhwiki.txt', 'r', encoding='utf-8'))
    model = Word2Vec(wiki_data, size=300, window=5, min_count=5, workers=multiprocessing.cpu_count())

    model.save('wiki_zh_cn.bin')
