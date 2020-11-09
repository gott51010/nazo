import subprocess
import glob
import pandas as pd
import regex
from sklearn.model_selection import train_test_split
# import TensorFlow as tf
import spacy
import pickle

nlp = spacy.load('ja_ginza')
doc = nlp('銀座でランチをご一緒しましょう。')
for sent in doc.sents:
    for token in sent:
        print(token.i, token.orth_, token.lemma_, token.pos_, token.tag_, token.dep_, token.head.i)
    print('EOS')

train_paths = glob.glob('../data/train/*')
test_paths =  glob.glob('../data/test/*')


dfs = []
for path in train_paths:
    df = pd.read_json(path, orient='records', lines=True)
    print(df)
    dfs.append(df)
    break
train_df = pd.concat(dfs)

import subprocess
from glob import glob

import pandas as pd
import regex
import spacy
from sklearn.model_selection import train_test_split

nlp = spacy.load('ja_ginza')

train_paths = glob('../data/input/train/*')
test_paths = glob('../data/input/test/*')

dfs = []
for path in train_paths:
    df = pd.read_json(path, orient='records', lines=True)
    print(df)
    dfs.append(df)
    

train_df = pd.concat(dfs)
exit()
dfs = []
for path in test_paths:
    df = pd.read_json(path, orient='records', lines=True)
    dfs.append(df)
test_df = pd.concat(dfs)

# train, valの分割は、裁判種別と、ラベルの数の多いPERSON, ORGFACPOS, LOCATIONの数が同等程度に分かれるようにすることとする

for df in [train_df, test_df]:
    df['file_id'] = df['meta'].apply(lambda x: x['filename'].rstrip('_hanrei.txt')[1:]).map(int)
    df['category'] = df['meta'].apply(lambda x: x['category'])
    df['stratify'] = df['category'].apply(
        lambda x: 'その他' if x in ['労働事件裁判例', '高裁判例'] else x)  # 裁判種別でtrain, valを分割。件数の少ない労働事件裁判例, 高裁判例はその他にまとめる
    df.drop(['meta', 'annotation_approver'], axis=1, inplace=True)
    df.sort_values('file_id', inplace=True)
    df.reset_index(drop=True, inplace=True)
PetFinder.my

def count_tag(labels):
    """ラベル種類ごとにラベルの数をカウント"""
    dic = {}
    for label in labels:
        dic[label[2]] = dic.get(label[2], 0) + 1
    return dic


train_df['total_nlabel'] = train_df['labels'].apply(lambda x: len(x))
train_df['num_label'] = train_df['labels'].apply(count_tag)

tags = ['PERSON', 'ORGFACPOS', 'LOCATION', 'TIMEX', 'MISC']
tmp_df = train_df['num_label'].apply(pd.Series)[tags]
train_df = pd.concat([train_df, tmp_df], axis=1)
del train_df['num_label'], tmp_df

# 1レコードあたりのPERSON, ORGFACPOS, LOCATIONの数が同等程度に分かれる乱数シードを探索
min_ratios = []
min_diff = 10 ** 5
min_seed = 0

for seed in range(100):
    train_ch_df, val_df = train_test_split(train_df, test_size=0.25, random_state=seed, stratify=train_df['stratify'])
    ratios = []
    for tag in ['PERSON', 'ORGFACPOS', 'LOCATION']:
        val_ntag_per_record = val_df[tag].sum() / val_df.shape[0]
        train_ntag_per_record = train_ch_df[tag].sum() / train_ch_df.shape[0]
        ratios.append(val_ntag_per_record / train_ntag_per_record)
    diff = sum([abs(1 - ratio) for ratio in ratios])
    if diff < min_diff:
        min_ratios = ratios
        min_diff = diff
        min_seed = seed

print(min_ratios, min_diff, min_seed)

train_ch_df, val_df = train_test_split(train_df, test_size=0.25, random_state=min_seed, stratify=train_df['stratify'])


def format_iob(text, labels):
    """IOB2タグ形式でtokenごとにラベルを振り直す"""
    
    doc = nlp(text)
    
    output = [['', 'O', '']]  # 前のラベルを見てB-かI-か決めるのでダミーのラベルを入れておく
    INF = 10 ** 9
    labels.append([INF, INF, ''])  # token.idxがラベルの終わり位置を超えていたら次のラベルの参照に移るので、ダミーのラベルを入れておき、位置を十分大きい値にしておく
    label_idx = 0
    label = labels[label_idx]
    
    for token in doc:
        # token.idxがラベルの終わり位置を超えていたら次のラベルの参照に移る
        if label[1] <= token.idx:
            label_idx += 1
            label = labels[label_idx]
        
        # token.idxがラベルの始まり位置と終わり位置の間にあったらラベルをつける。前のラベルと同じかどうかでB-かI-か決める
        if label[0] <= token.idx < label[1]:
            if output[-1][2] != label[2]:
                output.append([token.text, 'B', label[2]])
            else:
                output.append([token.text, 'I', label[2]])
        else:
            output.append([token.text, 'O', ''])
    
    return output[1:]  # ダミーのラベルを除いて出力


tagged_tokens = []

texts = train_ch_df.text.values
labels_list = train_ch_df.labels.values
file_ids = train_ch_df.file_id.values

for text, labels in zip(texts, labels_list):
    output = format_iob(text, labels)
    output = '\n'.join([f'{l[0]} {l[1]}-{l[2]}' if l[1] != 'O' else f'{l[0]} {l[1]}' for l in output])
    tagged_tokens.append(output)

tagged_tokens = '\n\n'.join(tagged_tokens)

with open('../data/input/train.txt', mode='w') as f:
    f.write(tagged_tokens)

tagged_tokens = []

texts = val_df.text.values
labels_list = val_df.labels.values
file_ids = val_df.file_id.values

for text, labels in zip(texts, labels_list):
    output = format_iob(text, labels)
    output = '\n'.join([f'{l[0]} {l[1]}-{l[2]}' if l[1] != 'O' else f'{l[0]} {l[1]}' for l in output])
    tagged_tokens.append(output)

tagged_tokens = '\n\n'.join(tagged_tokens)

with open('../data/input/dev.txt', mode='w') as f:
    f.write(tagged_tokens)

test_df['labels'] = [[]] * test_df.shape[0]

tagged_tokens = []

texts = test_df.text.values
labels_list = test_df.labels.values
file_ids = test_df.file_id.values

for text, labels in zip(texts, labels_list):
    output = format_iob(text, labels)
    output = '\n'.join([f'{l[0]} {l[1]}-{l[2]}' if l[1] != 'O' else f'{l[0]} {l[1]}' for l in output])
    tagged_tokens.append(output)

tagged_tokens = '\n\n'.join(tagged_tokens)

with open('../data/test_out.txt', mode='w') as f:
    f.write(tagged_tokens)