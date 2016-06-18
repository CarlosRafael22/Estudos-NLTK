import nltk
import random
from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier


from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI #inhirate from Classifier class
from statistics import mode
## So pra contar o tempo gasto em cada execucao
import time
import pickle

import re
from nltk.tokenize import TweetTokenizer

########################################################

tknzr = TweetTokenizer()

def reduce_tweets_words():

	#Ja vai pegar os tweets dos txts e tokeniza-los e colocar nesse array como tuplas
	# (Twitter_tokenizado , categoria_twitter)
	# ([u'RT', u'@mpvine', u':', u'If', u'fifty', u'million', u'people', u'say', u'a', 
	# u'foolish', u'thing', u',', u"it's", u'still', u'a', u'foolish', u'thing', u'.'], 'pos')
	tokenized_tweets = []
	with open("StayTweets1.txt") as pos:
		lines = pos.readlines()
		for r in lines:
			tokenized_tweets.append((tknzr.tokenize(r), "pos"))

	with open("LeaveTweets1.txt") as neg:
		lines = neg.readlines()
		for r in lines:
			tokenized_tweets.append((tknzr.tokenize(r), "neg"))

	all_words = []
	print(tokenized_tweets[3])

	#############################################################################
	#
	# AGORA QUE OS TWEETS ESTAO EM TUPLAS (tweets_tokenizados , categoria) VAMOS
	# REDUZIR O TAMANHO DOS TWEETS TOKENIZADOS TIRANDO STOPWORDS E OUTRAS COISAS
	# QUE NAO AGREGAM NA EXTRACAO DE CARACTERISTICAS
	#
	#############################################################################

	# Fazendo o stopwords nas palavras dos documentos pra tirar muita coisa inutil
	stop_words = set(stopwords.words("english"))
	# print(stop_words)
	# print('\n')

	#Vou tentar melhorar a lista de stopwords colocando nela algumas pontuacoes q nao servem de nada
	# Fiz elas com unicode pq eh assim que as stop_words estao
	punctuation = [u'.', u'-', u',', u'"', u'(', u')', u':', u'?', u"'", u'--', u';', u'!', u'$', u'*']
	punctuation = set(punctuation)
	new_stop_words = stop_words.union(punctuation)

	twitter_symbols = [u'RT']
	twitter_symbols = set(twitter_symbols)
	new_stop_words = new_stop_words.union(twitter_symbols)

	# NA VERDADE NAO TO CONSEGUINDO TIRAR O @USER DO RT MAS ISSO
	# NAO VAI INTERFERIR POIS A FREQUENCIA DE SE TER UM @USER DO MESMO USER EH POUCA
	user_rt_pattern = "@\w+?"
	url_pattern = 'http[s]:/'

	match = re.match(user_rt_pattern, '@mpvine alo filho')
	#print(match.group())

	
	# filtered_tweets = [for tweet_cat in tokenized_tweets if ]
	filtered_tweets = []
	tokens_to_be_removed = []
	#Para cada tupla (tweet_tok,categoria)
	for tweet_cat in tokenized_tweets[:3]:
		print(tweet_cat)
		print('\n')
		#Pra cada token desse tweet
		for token in tweet_cat[0]:
			#Se o token for uma das stop_words ou ter o Regex de URl ou RT a gnt tira
			print(token)
			if token in new_stop_words or re.match(url_pattern, token) or re.search(user_rt_pattern, token):
				tokens_to_be_removed.append(token)
				print(tokens_to_be_removed)

		#Vi todos os tokens q eram pra ser removidos desse tweet
		#Agora vou remove-los
		for token in tokens_to_be_removed:
			tweet_cat[0].remove(token)
		print('\n')
		print(tweet_cat[0])
		print('\n')

		#Limpar o tokens_to_be_removed pq senao vai sempre acumular de outros tweets
		tokens_to_be_removed = []
		#Adiciona o tweet sem as stopwords na nova lista
		filtered_tweets.append(tweet_cat)

	# Exemplo de tweet filtrado com stopwords
	# ([u'@mpvine', u'If', u'fifty', u'million', u'people', u'say',
	# u'foolish', u'thing', u"it's", u'still', u'foolish', u'thing'], 'pos')
	print(filtered_tweets)
	# print(filtered_tweets[6])
	# print(filtered_tweets[38])


	#######################################################################
	#
	#
	#
	#######################################################################


	#Arquivo com as novas tuplas dos tweets filtrados
	# with open('FilteredTweets.txt', 'w') as outfile:
	# 	for item in filtered_tweets:
 #  			outfile.write("%s\n" % item)


	#Tirando todas as stopwords dos tweets
	#all_tweets_no_stopwords = [w for w in documents if w not in new_stop_words]

	#Vou transformar all_words em Set pq ai a ordem nao importa e 
	#tem como fazer o JOIN e deixar todas as palavras em um mesmo array
	# all_words = set()
	# for line in all_lines:
	# 	line_tokenized = set(word_tokenize(line))
	# 	all_words.union(line_tokenized)
	# print(len(all_words))

	# all_words = nltk.FreqDist(all_words)
	# return all_words

reduce_tweets_words()
