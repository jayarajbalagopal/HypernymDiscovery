import os
import nltk
from tqdm import tqdm
import re
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from nltk.corpus import wordnet
import argparse
import numpy as np


def save(filename, data):
	with open(filename, "wb") as f:
	    pickle.dump(data, f)

def load(filename):
	with open(filename, "rb") as f:
	    data = pickle.load(f)
	return data

def load_nltk():
	nltk.download('punkt')
	nltk.download('wordnet')
	nltk.download('averaged_perceptron_tagger')

def clean_sentence(sentence):
    sentence = sentence.strip()
    sentence = ''.join(ch if ch.isalnum() else ' ' for ch in sentence)
    sentence = ' '.join(sentence.split())
    sentence = sentence.lower()
    return sentence

def extract_sentences(webbase_path):
	sentences = []
	for root, dirs, files in os.walk(webbase_path):
		for file in files:
		    if file.startswith('weather'): # Filter News
		        with open(os.path.join(root, file), 'r') as f:
		            text = f.read()
		            text_sentences = nltk.sent_tokenize(text)
		            for sentence in tqdm(text_sentences):
		              sentences.append(clean_sentence(sentence))
	return sentences

def build_vocab(sentence):
	vectorizer = CountVectorizer()
	vectorizer.fit(sentences)
	vocab = vectorizer.vocabulary_
	return vocab

def generate_pairs(vocab):
	wordnet = nltk.corpus.wordnet
	pos_tags = nltk.pos_tag(vocab)
	candidate_hypernyms = [word for word, tag in pos_tags if tag.startswith('N')]

	hypernym_hyponym_pairs = []
	for hypernym in tqdm(candidate_hypernyms):
		hyponyms = set()
		synsets = wordnet.synsets(hypernym)
		for synset in synsets:
			for hyponym in synset.hyponyms():
				hyponyms.update(hyponym.lemma_names())
		hypernym_hyponym_pairs.append((hypernym, hyponyms))

	for i in range(len(hypernym_hyponym_pairs)):
	    invalid_hyponyms = []
	    for hyponym in hypernym_hyponym_pairs[i][1]:
	        if hyponym not in vocab.keys():
	            invalid_hyponyms.append(hyponym)
	    for hyponym in invalid_hyponyms:
	        hypernym_hyponym_pairs[i][1].remove(hyponym)
	return hypernym_hyponym_pairs

def build_model(sentences, vocab):
	tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in tqdm(sentences)]
	w2v_model = Word2Vec(tokenized_sentences, min_count=1)
	return w2v_model

def find_possible_hypernyms(query_word, hypernym_hyponym_pairs, model):
    possible_hypernyms = {}
    for word, similarity in model.wv.most_similar(query_word):
        for hypernym, hyponyms in hypernym_hyponym_pairs:
            if word in hyponyms:
                if hypernym not in possible_hypernyms:
                    possible_hypernyms[hypernym] = similarity
                else:
                    possible_hypernyms[hypernym] += similarity
    return sorted(possible_hypernyms.items(), key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--build', action="store_true")
	build = parser.parse_args().build

	if build:
		load_nltk()
		sentences = extract_sentences("data")
		vocab = build_vocab(sentences)
		hypernym_hyponym_pairs = generate_pairs(vocab)
		w2v_model = build_model(sentences, vocab)
		VOC = "vocab.pkl"
		save(VOC, vocab)
		PIK = "pairs.pkl"
		save(PIK, hypernym_hyponym_pairs)
		MOD = "model.pt"
		save(MOD, w2v_model)
	else:
		VOC = "vocab.pkl"
		PIK = "pairs.pkl"
		MOD = "model.pt"
		hypernym_hyponym_pairs = load(PIK)
		vocab = load(VOC)
		w2v_model = load(MOD)

	word = input("Enter Query : ")
	while word not in vocab:
		print("Error! word not in vocab")
		word = input("Enter Query : ")
	hypernyms = find_possible_hypernyms(word, hypernym_hyponym_pairs, w2v_model)
	print(hypernyms)