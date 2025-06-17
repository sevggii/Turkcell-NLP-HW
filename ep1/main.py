# NLP nedir?

# NLP ile neler yapıyoruz? => 
# Metin Sınıflandırma -> E posta spam mı değil mi?
# Duygu Analizi -> (mutlu mu ? üzgün mü?)
# Özetleme -> 
# Metin Üretimi -> 
# Chatbot 
# Named Entity Recognition



# Bölüm 1 Müfredat Konuları


# Kütüphaneler -> numpy/pandas 
# NLTK => Temel nlp işlemleri yapan.
# scikit-learn


import nltk 
nltk.download('punkt_tab') # punkt_tab => Tokenizer

text = "Natural Language Processing is a branch of artificial intelligence."

# Tokenization
from nltk.tokenize import word_tokenize

tokens = word_tokenize(text)
print(tokens)
#

# Stop-Word Removal
# is,the,on,at,in
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english')) #Dosyadaki kelimeleri oku.
filtered_tokens = [word for word in tokens if word not in stop_words]
print(filtered_tokens)
#

# Lemmatization -> Kök haline getirme
# running -> run
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
# v =>verb -> fiil
# n =>noun -> isim
# a => adjective -> sıfat
# r => adverb (zarf)
print(lemmatizer.lemmatize('running', pos='n'))


# Pos tagging => Part of Speech Tagging
nltk.download('averaged_perceptron_tagger_eng')
from nltk import pos_tag

pos_tags = pos_tag(filtered_tokens)
print(pos_tags)
#


# NER => Named Entity Recognition
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

from nltk import ne_chunk
tree = ne_chunk(pos_tags)
print(tree)
#


# You have chosen
# YoU hAvE ChOSen

# Metin temizleme ve ön işleme 
# Lowercasing
text = "Natural Language Processing is, a branch of artificial intelligence. %100"

text = text.lower()
print(text)
#

# Remove Punctuation
import re
text = re.sub(r'[^\w\s]', '', text) #Regex => Regular Expression
print(text)
#

#
text = re.sub(r'\d+', '', text)
print(text)
#


# Vectorize Etmek

# Bag Of Words
corpus = [
    "Natural Language Processing is a branch of artificial intelligence.",
    "I love studying NLP.",
    "Language is a tool for communication.",
    "Language models can understand texts."
]
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
print(X.toarray())
#

# Tf-Idf -> Term Frequency - Inverse Document Frequency

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer2 = TfidfVectorizer()
X2 = vectorizer2.fit_transform(corpus)

print(vectorizer2.get_feature_names_out())
print(X2.toarray())


# Fonkisyon 

# pipeline => 
# 1-Tokenization - lowercasing 
# 2- Stopwords Temizliği
# 3- Lemmatization
# 4- TF-IDF Vektörleştirme
# 5- Feature isimlerini ve arrayi ekrana yazdır.


# generate a corpus of 10 about AI in english
corpus = [
    "Artificial Intelligence is the future.",
    "AI is changing the world.",
    "AI is a branch of computer science.",
    "Machine learning is a subset of AI.",
    "Deep learning enables machines to learn from data.",
    "AI can improve healthcare and education.",
    "Natural Language Processing allows machines to understand text.",
    "Self-driving cars use AI to navigate.",
    "AI systems can recognize speech and images.",
    "Ethics in AI is becoming more important every day."
]

import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Gerekli verileri indir
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def nlp_pipeline(corpus):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    cleaned_corpus = []

    for text in corpus:
        # Lowercasing
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Tokenization
        tokens = word_tokenize(text)
        # Stopword removal + sadece harf içeren kelimeleri al
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        # Lemmatization
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        # Cümleyi tekrar string'e çevir
        cleaned_text = ' '.join(tokens)
        cleaned_corpus.append(cleaned_text)

    # TF-IDF vektörleştirme
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(cleaned_corpus)

    # Feature isimlerini ve array'i yazdır
    print("Feature Names:\n", vectorizer.get_feature_names_out())
    print("\nTF-IDF Array:\n", X.toarray())

# Fonksiyonu çağır
nlp_pipeline(corpus)



#





