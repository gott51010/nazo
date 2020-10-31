import subprocess
from glob import glob
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