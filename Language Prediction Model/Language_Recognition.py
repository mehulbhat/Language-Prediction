import matplotlib
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np
import string

from collections import defaultdict

from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

import joblib
import pickle as pkl

from helper_code import *

model = joblib.load('Data/Models/final_model.joblib')
vectorizer = joblib.load('Data/Vectorizers/final_model.joblib')
def open_file(filename):
    with open(filename, 'r') as f:
        data = f.readlines()
    return data

data_raw = dict()
data_raw['sk'] = open_file('Data/Sentences/train_sentences.sk')
data_raw['cs'] = open_file('Data/Sentences/train_sentences.cs')
data_raw['en'] = open_file('Data/Sentences/train_sentences.en')

def show_statistics(data):
    for language, sentences in data.items():
        
        number_of_sentences = 0
        number_of_words = 0
        number_of_unique_words = 0
        sample_extract = ''
        
        word_list = ' '.join(sentences).split()
        
        number_of_sentences = len(sentences)
        number_of_words = len(word_list)
        number_of_unique_words = len(set(word_list))
        sample_extract = ' '.join(sentences[0].split()[0:7])
        
        print(f'Language: {language}')
        print('-----------------------')
        print(f'Number of sentences\t:\t {number_of_sentences}')
        print(f'Number of words\t\t:\t {number_of_words}')
        print(f'Number of unique words\t:\t {number_of_unique_words}')
        print(f'Sample extract\t\t:\t {sample_extract}...\n')

#Visualizing the raw data (number of sentences,words,etc.)
show_statistics(data_raw)

do_law_of_zipf(data_raw)

def preprocess(text):
    '''
    Removes punctuation and digits from a string, and converts all characters to lowercase. 
    Also clears all \n and hyphens (splits hyphenated words into two words).
    
    '''
        
    preprocessed_text = text
    preprocesses_text = text.lower().replace('-',' ')
    
    translation_table = str.maketrans('\n',' ',string.punctuation+string.digits)
    
    preprocessed_text = preprocessed_text.translate(translation_table)
    
    return preprocessed_text

data_prepropressed = {k: [preprocess(sentence) for sentence in v ] for k,v in data_raw.items()}

#Visualizing the raw and preprocessed data
print('Raw')
show_statistics(data_raw)
print('\nPreprocessed')
show_statistics(data_prepropressed)

sentences_train, y_train = [], []

for k, v in data_prepropressed.items():
    for sentence in v:
        sentences_train.append(sentence)
        y_train.append(k)
        
vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(sentences_train)

naive_classifier = MultinomialNB()
naive_classifier.fit(X_train, y_train)

data_val = dict()
data_val['sk'] = open_file('Data/Sentences/val_sentences.sk')
data_val['cs'] = open_file('Data/Sentences/val_sentences.cs')
data_val['en'] = open_file('Data/Sentences/val_sentences.en')

data_val_pre = {k: [preprocess(sentence) for sentence in v] for k, v in data_val.items()}

sentences_val, y_val = [], []

for k, v in data_val_pre.items():
    for sentence in v:
        sentences_val.append(sentence)
        y_val.append(k)
        
X_val = vectorizer.transform(sentences_val)
predictions = naive_classifier.predict(X_val)

#visualizing the confusion matrix
plot_confusion_matrix(y_val, predictions, ['sk', 'cs', 'en'])

#checking the accuracy with F1 score
f1_score(y_val, predictions, average = 'weighted')

#Training the model again with alpha parameter for better accuracy
naive_classifier = MultinomialNB(alpha=0.0001, fit_prior = False)
naive_classifier.fit(X_train, y_train)

predictions = naive_classifier.predict(X_val)

plot_confusion_matrix(y_val, predictions, ['sk', 'cs', 'en'])

#improved F1 score
f1_score(y_val, predictions, average = 'weighted')

#using the concept of subwords for better perspective

# taken from https://arxiv.org/abs/1508.07909

import re, collections
def get_stats(vocab):
    pairs = collections.defaultdict(int) 
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq 
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word] 
    return v_out
def get_vocab(data):

    words = []
    for sentence in data:
        words.extend(sentence.split())
        
    vocab = defaultdict(int)
    for word in words:
        vocab[' '.join(word)] += 1
        
    return vocab
vocab = get_vocab(sentences_train)
# also taken from original paper
for i in range(100):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get) 
    vocab = merge_vocab(best, vocab)
merges = defaultdict(int)
for k, v in vocab.items():
    for subword in k.split():
        if len(subword) >= 2:
            merges[subword] += v
merge_ordered = sorted(merges, key=merges.get, reverse=True)
pkl.dump(merge_ordered, open('Data/Auxiliary/merge_ordered.pkl', 'wb'))
def split_into_subwords(text):
    merges = pkl.load(open('Data/Auxiliary/merge_ordered.pkl', 'rb'))
    subwords = []
    for word in text.split():
        for subword in merges:
            subword_count = word.count(subword)
            if subword_count > 0:
                word = word.replace(subword, ' ')
                subwords.extend([subword]*subword_count)
    return ' '.join(subwords)

#Checking how the function works
split_into_subwords('hello my name is Jason')

data_prepropressed_subwords = {k: [split_into_subwords(sentence) for sentence in v] for k, v in data_prepropressed.items()}

#visualizing data with sub-words in consideration
show_statistics(data_prepropressed_subwords)

data_train_subwords = []
for sentence in sentences_train:
    data_train_subwords.append(split_into_subwords(sentence))

data_val_subwords = []
for sentence in sentences_val:
    data_val_subwords.append(split_into_subwords(sentence))

vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(data_train_subwords)
X_val = vectorizer.transform(data_val_subwords)

naive_classifier = MultinomialNB(alpha = 1.0, fit_prior = False)
naive_classifier.fit(X_train, y_train)

predictions = naive_classifier.predict(X_val)

#seeing the final confusion matrix (final trained model accuracy)
plot_confusion_matrix(y_val, predictions, ['sk', 'cs', 'en'])

f1_score(y_val, predictions, average='weighted')

#Use this function for Predicting language from your own data
text = 'okrem iného ako durič na brlohárenie'
text = preprocess_function(text)
text = [split_into_subwords_function(text)]
text_vectorized = vectorizer.transform(text)
model.predict(text_vectorized)

joblib.dump(naive_classifier, 'Data/Models/final_model.joblib')
joblib.dump(vectorizer, 'Data/Vectorizers/final_model.joblib')
