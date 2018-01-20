import jieba

text = '流浪汉听到声音，抬起头，看着我们傻笑，我吓得缩回脑袋。很多孩子，都在放肆地笑，汤姆却没笑，她下了去跑回家，拿出几袋雪饼。看着汤姆向流浪汉走去，我紧张得不敢说话，大孩子们也不敢笑了。汤姆小跑了几步，追上流浪汉，那张黢黑而肮脏的脸转了过来。汤姆将雪饼递出去，那人看起来有些害怕，接之前，还犹豫了一下。'

chout1 = jieba.lcut(text, cut_all=False)
print(len(chout1), chout1[0:])

chout3 = jieba.lcut(text, cut_all=True, HMM=True)
print(len(chout3), chout3[0:])
