#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 13:47:12 2020

@author: prakhar
"""

from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentLSTMEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from pathlib import Path
import pandas as pd
"""
data = pd.read_csv("/home/prakhar/Documents/iiitb/intel/data/emoji_train.csv", encoding='latin-1').sample(frac=1).drop_duplicates()


data = data[['classes', 'title']].rename(columns={"classes":"label", "title":"text"})

data.head()


data['label'] = '__label__' +data['label'].astype(str)
data.head()

data['text'] = data['text'].apply(lambda k: k.lower().strip())
data.head()
data.iloc[0:int(len(data)*0.8)].to_csv('/home/prakhar/Documents/iiitb/intel/data/train.csv', sep='\t', index = False, header = False)
data.iloc[int(len(data)*0.8):int(len(data)*0.9)].to_csv('/home/prakhar/Documents/iiitb/intel/data/test.csv', sep='\t', index = False, header = False)
data.iloc[int(len(data)*0.9):].to_csv('/home/prakhar/Documents/iiitb/intel/data/dev.csv', sep='\t', index = False, header = False);


corpus = NLPTaskDataFetcher.load_classification_corpus(Path('./data'), test_file='test.csv', dev_file='dev.csv', train_file='train.csv')

word_embeddings = [
					WordEmbeddings('glove'),
					FlairEmbeddings('news-forward'),
					FlairEmbeddings('news-backward')
				]

document_embeddings = DocumentLSTMEmbeddings(word_embeddings, hidden_size=512, reproject_words=True, reproject_words_dimension=256)

classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(), multi_label=False)

trainer = ModelTrainer(classifier, corpus)

trainer.train('./data', max_epochs=10)

"""
from flair.models import TextClassifier
from flair.data import Sentence

classifier = TextClassifier.load('./data/best-model.pt')

sentence = Sentence('i love you')

classifier.predict(sentence)

print(sentence.labels)

