
# 参考文章
http://sqrtf.com/chinese-word2vec-embedding-in-keras/

#
仅支持python3
#
依赖 gensim word2vec opencc-python(https://github.com/yichen0831/opencc-python)


1、将bz2 xml 文件转换为txt文件
~~~shell
python process_wiki.py zhwiki-latest-pages-articles.xml.bz2 zh_tw_zhwiki.txt
~~~
2、生成 Word2Vec 字典文件
~~~shell
python save_mode.py
~~~