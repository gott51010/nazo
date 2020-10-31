# encoding=utf-8
import jieba
import jieba.analyse

txt = "狂人日记某君昆仲，今隐其名，皆余昔日在中学时良友；分隔多年，消息渐阙。日前偶闻其一大病；适归故乡，迂" \
      "道往访，则仅晤一人，言病者其弟也。劳君远道来视，然已早愈，赴某地候补矣。因大笑，出示日记二册，谓可见当日病状，" \
      "不妨献诸旧友。持归阅一过，知所患盖“迫害狂”之类。语颇错杂无伦次，又多荒唐之言；亦不著月日，惟墨色字体不一，知非一时所书。间" \
      "亦有略具联络者，今撮录一篇，以供医家研究。记中语误，一字不易；惟人名虽皆村人，不为世间所知，无关大体，然亦悉易去。至于书名，则本" \
      "人愈后所题，不复改也。七年四月二日识。一今天晚上，很好的月光。我不见他，已是三十多年；今天见了，精神分外爽快。才知道以前的三十多年" \
      "，全是发昏；然而须十分小心。不然，那赵家的狗，何以看我两眼呢？"

seg_list = jieba.cut(txt, cut_all=False)  # 默认是精确模式
print("\t".join(seg_list))
re = jieba.analyse.extract_tags(txt, topK=20)

print(re)
words = jieba.lcut(txt)  # 使用精确模式对文本进行分词
counts = {}  # 通过键值对的形式存储词语及其出现的次数

for word in words:
    if len(word) == 1:  # 单个词语不计算在内
        continue
    else:
        counts[word] = counts.get(word, 0) + 1  # 遍历所有词语，每出现一次其对应的值加 1

items = list(counts.items())  # 将键值对转换成列表
items.sort(key=lambda x: x[1], reverse=True)  # 根据词语出现的次数进行从大到小排序

for i in range(15):
    word, count = items[i]
    print("{0:<5}{1:>5}".format(word, count))
