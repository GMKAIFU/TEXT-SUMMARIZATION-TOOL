import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = sent_tokenize(text)
    return sentences

def get_sentence_similarity(sent1_tokens, sent2_tokens):
    stop_words = set(stopwords.words('english'))
    words1 = [word.lower() for word in sent1_tokens if word.isalnum() and word.lower() not in stop_words]
    words2 = [word.lower() for word in sent2_tokens if word.isalnum() and word.lower() not in stop_words]
    set1 = set(words1)
    set2 = set(words2)
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0

def build_similarity_matrix_custom(sentences_tokens):
    num_sentences = len(sentences_tokens)
    similarity_matrix = np.zeros((num_sentences, num_sentences))
    for i in range(num_sentences):
        for j in range(num_sentences):
            if i == j:
                continue
            similarity_matrix[i][j] = get_sentence_similarity(sentences_tokens[i], sentences_tokens[j])
    return similarity_matrix

def build_similarity_matrix_tfidf(sentences):
    if not sentences:
        return np.array([])
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        return similarity_matrix
    except ValueError:
        return np.zeros((len(sentences), len(sentences)))

def textrank_summarizer(text, num_sentences=3, use_tfidf_similarity=True):
    if not text.strip():
        return "Input text is empty."
    original_sentences = preprocess_text(text)
    if not original_sentences:
        return "No sentences found in the text."
    if len(original_sentences) <= num_sentences:
        return " ".join(original_sentences)

    if not use_tfidf_similarity:
        sentences_tokens = [word_tokenize(sentence) for sentence in original_sentences]
        similarity_matrix = build_similarity_matrix_custom(sentences_tokens)
    else:
        similarity_matrix = build_similarity_matrix_tfidf(original_sentences)

    if similarity_matrix.size == 0:
         return "Could not build similarity matrix (e.g., all sentences are stop words)."

    sentence_graph = nx.from_numpy_array(similarity_matrix)
    try:
        scores = nx.pagerank(sentence_graph, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        print("Warning: PageRank did not converge. Using equal scores.")
        scores = {i: 1.0/len(original_sentences) for i in range(len(original_sentences))}

    ranked_sentences_indices = sorted(scores, key=scores.get, reverse=True)
    top_sentence_indices = sorted(ranked_sentences_indices[:num_sentences])
    summary = " ".join([original_sentences[i] for i in top_sentence_indices])
    return summary

if __name__ == "__main__":
    # Make sure NLTK data is downloaded before running:
    # nltk.download('punkt')
    # nltk.download('stopwords')

    print("Welcome to the TextRank Summarizer!")
    print("Paste or type your article below. Type 'exit' (without quotes) to quit.")

    while True:
        print("\nEnter your article text:")
        article = []
        print("(Type 'END' alone on a line when done)")
        while True:
            line = input()
            if line.strip().upper() == 'END':
                break
            if line.strip().lower() == 'exit':
                print("Exiting summarizer.")
                exit()
            article.append(line)
        article_text = "\n".join(article).strip()
        if not article_text:
            print("No text entered. Try again or type 'exit' to quit.")
            continue

        summary = textrank_summarizer(article_text, num_sentences=3, use_tfidf_similarity=True)
        print("\nSummary:")
        print(summary)
        print("-" * 50)
