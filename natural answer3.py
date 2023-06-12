

pip install nltk gensim


import csv
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models

# Initialize NLTK stopwords
stop_words = set(stopwords.words('english'))


# Read the text data from the CSV file
with open('pdf_text.csv', 'r', encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file)
    next(reader)  
    text = ' '.join([row[0] for row in reader])


text = re.sub(r'\d+', '', text)  # Remove numbers
text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
text = text.lower()  # Convert to lowercase
tokens = word_tokenize(text)  # Tokenize the text
tokens = [token for token in tokens if token.isalpha()]  # Remove non-alphabetic tokens
tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords


]dictionary = corpora.Dictionary([tokens])

doc_term_matrix = [dictionary.doc2bow(tokens)]

tfidf = models.TfidfModel(doc_term_matrix)

top_keywords = tfidf[doc_term_matrix[0]]
top_keywords = sorted(top_keywords, key=lambda x: x[1], reverse=True)



lda_model = models.LdaModel(doc_term_matrix, num_topics=5, id2word=dictionary)

keyword_topics = lda_model.get_document_topics(doc_term_matrix[0])

for keyword, topic in zip(top_keywords, keyword_topics):
    print(f"Keyword: {dictionary[keyword[0]]}\t Topic: {topic[0]}")








