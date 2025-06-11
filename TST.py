# Import necessary libraries
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Preprocessing: Clean the text and split it into sentences
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    sentences = sent_tokenize(text)  # Split text into sentences
    return sentences

# Calculate Jaccard similarity between two tokenized sentences
def get_sentence_similarity(sent1_tokens, sent2_tokens):
    stop_words = set(stopwords.words('english'))  # Get English stop words

    # Filter out stopwords and non-alphanumeric words
    words1 = [word.lower() for word in sent1_tokens if word.isalnum() and word.lower() not in stop_words]
    words2 = [word.lower() for word in sent2_tokens if word.isalnum() and word.lower() not in stop_words]

    set1 = set(words1)
    set2 = set(words2)

    if not set1 or not set2:
        return 0.0  # If either sentence has no valid words, similarity is 0

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection / union if union != 0 else 0.0

# Build a similarity matrix using custom Jaccard similarity
def build_similarity_matrix_custom(sentences_tokens):
    num_sentences = len(sentences_tokens)
    similarity_matrix = np.zeros((num_sentences, num_sentences))  # Initialize matrix

    for i in range(num_sentences):
        for j in range(num_sentences):
            if i == j:
                continue  # Skip self-similarity
            similarity_matrix[i][j] = get_sentence_similarity(sentences_tokens[i], sentences_tokens[j])

    return similarity_matrix

# Build a similarity matrix using TF-IDF cosine similarity
def build_similarity_matrix_tfidf(sentences):
    if not sentences:
        return np.array([])  # Return empty matrix if no sentences provided

    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))

    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        return similarity_matrix
    except ValueError:
        # If TF-IDF fails (e.g., all sentences are stop words), return zero matrix
        return np.zeros((len(sentences), len(sentences)))

# Main TextRank summarizer function
def textrank_summarizer(text, num_sentences=3, use_tfidf_similarity=True):
    if not text.strip():
        return "Input text is empty."  # Handle empty input

    original_sentences = preprocess_text(text)

    if not original_sentences:
        return "No sentences found in the text."  # Handle no sentences case

    if len(original_sentences) <= num_sentences:
        return " ".join(original_sentences)  # Return all sentences if text is too short

    if not use_tfidf_similarity:
        # Use custom similarity if TF-IDF is not selected
        sentences_tokens = [word_tokenize(sentence) for sentence in original_sentences]
        similarity_matrix = build_similarity_matrix_custom(sentences_tokens)
    else:
        # Use TF-IDF similarity
        similarity_matrix = build_similarity_matrix_tfidf(original_sentences)

    if similarity_matrix.size == 0:
        return "Could not build similarity matrix (e.g., all sentences are stop words)."

    # Create a graph from the similarity matrix
    sentence_graph = nx.from_numpy_array(similarity_matrix)

    try:
        # Apply PageRank to score sentences
        scores = nx.pagerank(sentence_graph, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        # Handle cases where PageRank doesn't converge
        print("Warning: PageRank did not converge. Using equal scores.")
        scores = {i: 1.0 / len(original_sentences) for i in range(len(original_sentences))}

    # Rank sentences based on their PageRank scores
    ranked_sentences_indices = sorted(scores, key=scores.get, reverse=True)
    top_sentence_indices = sorted(ranked_sentences_indices[:num_sentences])  # Sort to maintain order

    # Generate summary using top ranked sentences
    summary = " ".join([original_sentences[i] for i in top_sentence_indices])
    return summary

# Interactive summarizer CLI
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

            if line.strip().upper() == 'END':  # Check if user has finished input
                break

            if line.strip().lower() == 'exit':  # Exit the program
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
