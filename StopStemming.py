from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer

#SIMPLE CLASS TO USE PRE-PROCESSING METHODS: STOPWORDS AND STEMMING


####### STOPWORDS PRE-PROCESSING

example_sentence = "This is an example sentence showing off stop word filtration."
stop_words = set(stopwords.words("english"))

#print(stop_words)

words = word_tokenize(example_sentence)
print(words)

#filtered_sentence = []
# for w in words:
# 	if w not in stop_words:
# 		filtered_sentence.append(w)
filtered_sentence = [w for w in words if not w in stop_words]

print(filtered_sentence)

example_text = "The bag-of-words model is a simplifying representation used in natural language processing and information retrieval (IR). In this model, a text (such as a sentence or a document) is represented as the bag (multiset) of its words, disregarding grammar and even word order but keeping multiplicity. Recently, the bag-of-words model has also been used for computer vision"
#print(sent_tokenize(example_text))
words_count = len(word_tokenize(example_text))
print(words_count)

#Vou botar filtrar com os stopwords no example_text
filtered_text = [w for w in word_tokenize(example_text) if w not in stop_words]
#print(filtered_text)
print(len(filtered_text))

#################################################################

############ STEMMING PRE-PROCESSING

ps = PorterStemmer()

new_text = "It is very important to be pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once."
words = word_tokenize(new_text)
for w in words:
	print(ps.stem(w))
