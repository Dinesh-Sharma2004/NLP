sentence="My name is Dinesh Sharma, and I am a 3rd year AIDS student! Shivam Vats is a Dumb student. I like to play badminton but Shivam is playing Cricket."

#     SENTENCE AND WORD TOKENIZATION    
    
    #1.Using Regular Expression
import re
word_tokens_re = re.findall(r"\b\w+\b", sentence)
sent_tokens_re = [s.strip() for s in re.split(r'[.!?]', sentence) if s.strip()]


print(word_tokens_re)
print(sent_tokens_re)


    #2. Using NLTK
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize

word_tokens_nltk=word_tokenize(sentence)
sent_tokens_nltk=sent_tokenize(sentence)

print(word_tokens_nltk)
print(sent_tokens_nltk)


    #3. Using Spacy
import spacy

nlp=spacy.load("en_core_web_sm")
doc=nlp(sentence)
token=[i for i in doc]
print(token)



#  STOP WORD REMOVAL  

#1. Using NLTK
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

stop_words=set(stopwords.words('english'))
tokens=word_tokenize(sentence.lower())

filtered_tokens=[word for word in tokens if word not in stop_words]

print("\nNLTK\n")
print("Original:", sentence)
print("Filtered:",filtered_tokens)

#2. Using Spacy
import spacy 
nlp=spacy.load("en_core_web_sm")
doc=nlp(sentence)
filtered_words=[token.text for token in doc if not token.is_stop]

print("\nSpacy\n")
print("Original:",sentence)
print("Filtered:",filtered_words)

#3. Gensim
from gensim.parsing.preprocessing import remove_stopwords
filtered_text=remove_stopwords(sentence)

print("\nGensim\n")
print("Original:",sentence)
print("Filtered:",filtered_text)



# STEMMING AND LEMMATIZATION #

#1. Lemmatization (Using NLTK)
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

lemmatizer = WordNetLemmatizer()
word = "running"
lemma = lemmatizer.lemmatize(word, get_wordnet_pos(word))
print(f"Lemmatized word: {lemma}")

#2. Stemming (Using NLTK)
import nltk
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
word = "running"
stem = stemmer.stem(word)
print(f"Stemmed word: {stem}")


#  CASE FOLDING AND PUNCTUATION REMOVAL  #


#1. Using NLTK
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

tokens = tokenizer.tokenize(sentence)

print("Removing Punctutation")
print("Original",sentence)
print("Removing Punctuation:",tokens)


