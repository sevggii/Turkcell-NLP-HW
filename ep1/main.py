import nltk
nltk.download('punkt_tab')  # punkt => Tokenizer

text = "Natural Language Processing is a branch of artificial intelligence."

from nltk.tokenize import word_tokenize

tokens=word_tokenize(text)
print(tokens)

##stopwords=tek başına anlamı olmayan, bağlaçlar gibi..

from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))  # Dosyadaki kelimeleri oku.
filtered_tokens = [word for word in tokens if word not in stop_words]
print(filtered_tokens)


##kelime köküne inme
# running -> run
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize('running', pos='v'))







