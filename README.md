# TEXT-SUMMARIZATION-TOOL

COMPANY: CODTECH IT SOLUTIONS

NAME: G M KAIFU

INTERN ID: CT06DN1863

DOMAIN: ARTIFICIAL INTELLIGENCE

DURATION: 6 WEEKS

MENTOR: NEELA SANTOSH

:

📝 Text Summarization Tool using NLP
A Python-based tool to automatically summarize lengthy articles using Natural Language Processing techniques.

📌 Project Objective
Task:
Create a tool that summarizes lengthy articles using Natural Language Processing (NLP) techniques.

Deliverable:
A Python script that accepts input text (such as articles, paragraphs, or documents) and returns a concise, meaningful summary.

💡 Project Description

This project presents a Text Summarization Tool developed in Python, designed to automatically generate concise summaries from lengthy input text using advanced Natural Language Processing (NLP) techniques. The goal of this tool is to assist users—students, researchers, content creators, and developers—by reducing large blocks of text into short, meaningful summaries that retain the most relevant information. This tool helps in saving time, improving comprehension, and enabling quick content analysis without reading the entire document.

At its core, the project implements the TextRank algorithm, which is a graph-based ranking algorithm inspired by Google’s PageRank. TextRank is used to extract the most significant sentences from a document based on their relative importance and similarity to each other. The method involves representing sentences as nodes in a graph and calculating the edges based on similarity scores. The higher the similarity between two sentences, the stronger their connection in the graph. PageRank is then applied to determine which sentences are most central and meaningful to the overall content.

The tool is designed for:

Students and researchers working on NLP projects.

Anyone needing automatic summarization for large documents.

Developers looking to integrate summarization into other applications.

🛠 Features:

📤 Accepts multi-line text input through the terminal.

🧠 Uses TextRank (based on PageRank) for ranking sentences.

⚙️ Two similarity models:

    TF-IDF-based Cosine Similarity (default and robust)

    Jaccard Custom Similarity (optional)
    
📈 Selects top-N sentences as the summary.

⛔ Handles short or stop-word-only inputs with clear messages.

🔁 Reusable loop interface — summarize multiple articles continuously.


 How to Run:
 
1. Install Required Libraries
   
     pip install nltk scikit-learn networkx numpy
   
3. Download NLTK Resources (only once)

     import nltk
   
     nltk.download('punkt')

     nltk.download('stopwords')


5. Run the Python Script

     python filename.py
   

📥 Input Instructions:

Paste or type your article into the terminal when prompted.

To finish typing your article: enter END on a new line.

To exit the tool anytime: type exit.


 Applications:
 
Research paper summarization

News article condensation

Blog digest creation

Automated content previews


👨‍💻 Author Purpose
This summarization tool is developed as part of a learning project to implement NLP techniques in real-world scenarios using Python. It demonstrates how to apply sentence tokenization, vectorization, similarity computation, and graph-based ranking for summarization.


# OUTPUT1

![Image](https://github.com/user-attachments/assets/a33247ae-47d0-4650-9272-8d13b7a946a8)

# OUTPUT2
![Image](https://github.com/user-attachments/assets/624514d8-445b-4f93-b5eb-89e911fcd455)
