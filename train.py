#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 13:47:12 2020

@author: prakhar
"""

import os
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings, ELMoEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from pathlib import Path

def segment_data(data_file):
    try:
        import pandas as pd
    except ImportError:
        raise
    
    data = pd.read_csv(data_file, encoding='latin-1').sample(frac=1).drop_duplicates()
    data = data[['classes', 'title']].rename(columns={"classes":"label", "title":"text"})
    data['label'] = '__label__' +data['label'].astype(str)
    data['text'] = data['text'].apply(lambda k: k.lower().strip())
    data.to_csv('./data/whole.csv', sep='\t', index = False, header = False)
    data.iloc[0:int(len(data)*0.8)].to_csv('./data/train.csv', sep='\t', index = False, header = False)
    data.iloc[int(len(data)*0.8):int(len(data)*0.9)].to_csv('./data/test.csv', sep='\t', index = False, header = False)
    data.iloc[int(len(data)*0.9):].to_csv('./data/dev.csv', sep='\t', index = False, header = False)
    return
    
def create_corpus():
    if os.path.exists('./data/train.csv') and  os.path.exists('./data/test.csv') and  os.path.exists('./data/dev.csv'):
        corpus = NLPTaskDataFetcher.load_classification_corpus(Path('./data'), test_file='test.csv', dev_file='dev.csv', train_file='train.csv')
        return corpus
    else: return 0
    
def initialize_embeddings():
    """
    Summary:
        Stacks the list of pre-trained embedding vectors to be used as word representation (in concat.)
    Return:
        list: Returns list of pretrained embeddings vectors
    """
    word_embeddings = [
			WordEmbeddings('glove'),
			FlairEmbeddings('news-forward'),
			FlairEmbeddings('news-backward'),
                        ELMoEmbeddings('medium')
		]
    return word_embeddings


if __name__ == '__main__':
    data_file = './data/emoji_train.csv'
    segment_data(data_file)
    corpus = create_corpus()

    if not corpus:
        raise

    word_embeddings = initialize_embeddings()
    document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=512, reproject_words=True, reproject_words_dimension=256, rnn_type='LSTM', rnn_layers=1, bidirectional=False)
    classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(), multi_label=False)
    trainer = ModelTrainer(classifier, corpus)
    trainer.train('./model', max_epochs=20, patience=5, mini_batch_size=32, learning_rate=0.1)

