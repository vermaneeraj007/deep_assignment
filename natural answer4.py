

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx


def preprocess_text(text):
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    words = [word_tokenize(sentence) for sentence in sentences]

    # Remove stopwords and punctuation, and perform stemming
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()
    preprocessed_words = [
        [ps.stem(word.lower()) for word in sentence if word.isalnum() and word.lower() not in stop_words]
        for sentence in words
    ]

    return preprocessed_words

def sentence_similarity(sent1, sent2):
    vec1 = sent1.reshape(1, -1)
    vec2 = sent2.reshape(1, -1)
    return 1 - cosine_distance(vec1, vec2)

def build_similarity_matrix(sentences):
    # Build a similarity matrix between sentences using sentence_similarity
    n = len(sentences)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j])

    return similarity_matrix

def text_rank_summarize(text, num_sentences):
    # Preprocess the text
    preprocessed_words = preprocess_text(text)

    word_frequencies = {}
    for sentence in preprocessed_words:
        for word in sentence:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

]    maximum_frequency = max(word_frequencies.values())
    for word in word_frequencies:
        word_frequencies[word] /= maximum_frequency

    # Create sentence vectors
    sentence_vectors = []
    for sentence in preprocessed_words:
        sentence_vector = np.zeros((len(word_frequencies),))
        for word in sentence:
            if word in word_frequencies:
                sentence_vector[list(word_frequencies).index(word)] = word_frequencies[word]
        sentence_vectors.append(sentence_vector)

    similarity_matrix = build_similarity_matrix(sentence_vectors)

    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i], sentence) for i, sentence in enumerate(sentences)), reverse=True)

    summary_sentences = [sentence for _, sentence in ranked_sentences[:num_sentences]]

    summary = ' '.join(summary_sentences)

    return summary


with open('text_file.txt', 'r', encoding='utf-8') as file:
   
