http://www.52nlp.cn/%E4%B8%AD%E8%8B%B1%E6%96%87%E7%BB%B4%E5%9F%BA%E7%99%BE%E7%A7%91%E8%AF%AD%E6%96%99%E4%B8%8A%E7%9A%84word2vec%E5%AE%9E%E9%AA%8C

# 参考文章
http://sqrtf.com/chinese-word2vec-embedding-in-keras/

依赖 gensim word2vec opencc-python(https://github.com/yichen0831/opencc-python)


1、将bz2 xml 文件转换为txt文件
~~~shell
python process_wiki.py zhwiki-latest-pages-articles.xml.bz2 zh_tw_zhwiki.txt
~~~
2、生成 Word2Vec 字典文件
~~~shell
python save_mode.py
~~~