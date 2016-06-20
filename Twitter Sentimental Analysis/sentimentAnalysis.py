# coding: utf-8
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
from nltk.tokenize import TweetTokenizer, word_tokenize

########################################################

tknzr = TweetTokenizer()
new_stop_words = set()

def reduce_tweets_words():

	#Ja vai pegar os tweets dos txts e tokeniza-los e colocar nesse array como tuplas
	# (Twitter_tokenizado , categoria_twitter)
	# ([u'RT', u'@mpvine', u':', u'If', u'fifty', u'million', u'people', u'say', u'a', 
	# u'foolish', u'thing', u',', u"it's", u'still', u'a', u'foolish', u'thing', u'.'], 'pos')
	tokenized_tweets = []

	# Tem 2853 no FeatureSet, sendo: 1286 Stay e 1567 Leave
	with open("StayTweets1.txt") as pos:
		lines = pos.readlines()
		print(len(lines))
		for l in lines:
			#Pra tirar se tiver emotions no formato /u2026 por exemplo
			l = l.decode('unicode_escape').encode('ascii','ignore')
			tokens = tknzr.tokenize(l)
			#Pega cada token e bota em minuscula
			lw_tokens = [w.lower() for w in tokens]
			tokenized_tweets.append((lw_tokens, "pos"))

	with open("StayTweetsDate.txt") as pos:
		lines = pos.readlines()
		print(len(lines))
		for l in lines:
			#Pra tirar se tiver emotions no formato /u2026 por exemplo
			l = l.decode('unicode_escape').encode('ascii','ignore')
			tokens = tknzr.tokenize(l)
			#Pega cada token e bota em minuscula
			lw_tokens = [w.lower() for w in tokens]
			tokenized_tweets.append((lw_tokens, "pos"))

	with open("StayTweetsDate2.txt") as pos:
		lines = pos.readlines()
		print(len(lines))
		for l in lines:
			#Pra tirar se tiver emotions no formato /u2026 por exemplo
			l = l.decode('unicode_escape').encode('ascii','ignore')
			tokens = tknzr.tokenize(l)
			#Pega cada token e bota em minuscula
			lw_tokens = [w.lower() for w in tokens]
			tokenized_tweets.append((lw_tokens, "pos"))

	###########################################################################################

	with open("LeaveTweets1.txt") as neg:
		lines = neg.readlines()
		#PQ TEM 1286 Stay e 1567 Leave AI DEIXA IGUAL
		lines = lines[:1286]
		print(len(lines))
		for l in lines:
			l = l.decode('unicode_escape').encode('ascii','ignore')
			tokens = tknzr.tokenize(l)
			#Pega cada token e bota em minuscula
			lw_tokens = [w.lower() for w in tokens]
			tokenized_tweets.append((lw_tokens, "neg"))

	with open("LeaveTweetsDate.txt") as neg:
		lines = neg.readlines()
		print(len(lines))
		for l in lines:
			l = l.decode('unicode_escape').encode('ascii','ignore')
			tokens = tknzr.tokenize(l)
			#Pega cada token e bota em minuscula
			lw_tokens = [w.lower() for w in tokens]
			tokenized_tweets.append((lw_tokens, "neg"))

	with open("LeaveTweetsDate2.txt") as neg:
		lines = neg.readlines()
		print(len(lines))
		for l in lines:
			l = l.decode('unicode_escape').encode('ascii','ignore')
			tokens = tknzr.tokenize(l)
			#Pega cada token e bota em minuscula
			lw_tokens = [w.lower() for w in tokens]
			tokenized_tweets.append((lw_tokens, "neg"))

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

	#Vou tentar melhorar a lista de stopwords colocando nela algumas pontuacoes q nao servem de nada
	# Fiz elas com unicode pq eh assim que as stop_words estao
	punctuation = [u'.', u'-', u',', u'"', u'(', u')', u':', u'?', u"'", u'--', u';', 
	u'!', u'$', u'*', u'&', u'...']
	punctuation = set(punctuation)
	global new_stop_words
	new_stop_words = stop_words.union(punctuation)

	twitter_symbols = [u'rt']
	twitter_symbols = set(twitter_symbols)
	new_stop_words = new_stop_words.union(twitter_symbols)

	# NA VERDADE NAO TO CONSEGUINDO TIRAR O @USER DO RT MAS ISSO
	# NAO VAI INTERFERIR POIS A FREQUENCIA DE SE TER UM @USER DO MESMO USER EH POUCA
	#user_rt_pattern = "@\w+?"
	#url_pattern = 'http[s]:/'
	emotions_pattern = '\u\d+'
	url_pattern = 'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'
	user_rt_pattern = '(?:@[\w_]+)'
    #user_rt_pattern = '(?:@[\w_]+)'

	match = re.match(emotions_pattern, 'asfasdf \u2026 alo filho')
	s = 'This is Some \u03c0 text that Has to be Cleaned\u2026! it\u0027s Annoying!'
	print(s.decode('unicode_escape').encode('ascii','ignore'))
	for w in s:
		w = w.lower()
	print(s)
	print('\n')
	
	# filtered_tweets = [for tweet_cat in tokenized_tweets if ]
	filtered_tweets = []
	tokens_to_be_removed = []
	print(tokenized_tweets[228:230])
	#Para cada tupla (tweet_tok,categoria)
	for tweet_cat in tokenized_tweets:
		#print(tweet_cat)
		#print('\n')

		#Pra cada token desse tweet
		for token in tweet_cat[0]:
			#Se o token for uma das stop_words ou ter o Regex de URl ou RT a gnt tira
			#print(token)
			if token in new_stop_words or re.match(url_pattern, token) or re.search(user_rt_pattern, token) or re.match(emotions_pattern, token):
				tokens_to_be_removed.append(token)
				#print(tokens_to_be_removed)

		#Vi todos os tokens q eram pra ser removidos desse tweet
		#Agora vou remove-los
		for token in tokens_to_be_removed:
			tweet_cat[0].remove(token)

		#print('\n')
		#print(tweet_cat[0])
		#print('\n')

		#Limpar o tokens_to_be_removed pq senao vai sempre acumular de outros tweets
		tokens_to_be_removed = []
		#Adiciona o tweet sem as stopwords na nova lista
		filtered_tweets.append(tweet_cat)

	# Exemplo de tweet filtrado com stopwords
	# ([u'@mpvine', u'If', u'fifty', u'million', u'people', u'say',
	# u'foolish', u'thing', u"it's", u'still', u'foolish', u'thing'], 'pos')
	print(filtered_tweets[228:230])


	#######################################################################
	#
	# JOGANDO TODOS OS TWEETS REDUZIDOS EM UM ARQUIVO
	#
	#######################################################################


	#Arquivo com as novas tuplas dos tweets filtrados
	with open('FilteredTweets2.txt', 'w') as outfile:
		for item in filtered_tweets:
  			outfile.write(str(item) + '\n')


  	return filtered_tweets


#############################################################################
# SO PRA REDUZIR UM NOVO TWEET QUE FORMOS CLASSIFICAR ISOLADAMENTE
def reduce_tweet(tweet_text):
	tweet_text = tweet_text.decode('unicode_escape').encode('ascii','ignore')
	text_tokenized = tknzr.tokenize(tweet_text)
	#Botando em minuscula pra nao ter diferenca
	text_tokenized = [w.lower() for w in text_tokenized]
	print(text_tokenized)

	#user_rt_pattern = "@\w+?"
	#url_pattern = 'http[s]:/'
	emotions_pattern = '\u\d+'
	url_pattern = 'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'
	user_rt_pattern = '(?:@[\w_]+)'
    #user_rt_pattern = '(?:@[\w_]+)'

	tokens_to_be_removed = []
	for token in text_tokenized:
		if token in new_stop_words or re.match(url_pattern, token) or re.search(user_rt_pattern, token) or re.match(emotions_pattern, token):
				tokens_to_be_removed.append(token)
				print(tokens_to_be_removed)

	#Vi todos os tokens q eram pra ser removidos desse tweet
	#Agora vou remove-los
	for token in tokens_to_be_removed:
		text_tokenized.remove(token)

	print('\n')
	print(text_tokenized)
	print('\n')

	return text_tokenized

##############################################################################

def getTop_tweet_words(filtered_tweets):

  	######################################################################
  	#
  	# AGORA QUE TEMOS TODOS OS TWEETS FILTRADOS COLOCAREMOS TODAS AS 
  	# PALAVRAS EM UMA LIST PRA SEREM AS FEATURES QUE IREMOS VERIFICAR
  	# AO TREINAR O CLASSIFICADOR
  	#
  	######################################################################

  	all_tweets_words = []

  	for tweet_cat in filtered_tweets:
  		for token in tweet_cat[0]:
  			all_tweets_words.append(token)

  	# Tem 20515 palavras nessa lista
  	print(len(all_tweets_words))
  	print(all_tweets_words[1230:1240])
  	print('\n')
  	all_tweets_words = nltk.FreqDist(all_tweets_words)
  	print(all_tweets_words.most_common(50))
  	print('\n')
  	top_tweets_words = all_tweets_words.most_common(2000)
	#print(top_tweets_words[1800:1805])


	#Como top_tweets_words retorna (word,freq) iremos pegar so as palavras que sao as keys
	top_tweets_features_keys = [wf[0] for wf in top_tweets_words]

	return top_tweets_features_keys



#Retorna uma lista com True ou False dizendo quais palavras da word_features o documento tem
# retorna: {u'even': True, u'story': False, u'also': True, u'see': True, u'much': False,.... }
def find_features(tweet):
	# Pega todas as palavras do documento e transforma em set pra retornar as palavras independente da frequencia dela
	tweet_words = set(tweet)
	# vai ser o dict dizendo quais palavras, de todas as tidas como mais importantes, estao presentes nese tweet
	features = {}
	#print(top_word_features_keys[:20])
	for w in top_tweets_features:
		features[w] = (w in tweet_words)
		# if(w in tweet_words):
		# 	print(w)

	return features

######################################################################################
#
# AGORA QUE TEMOS OS TWEETS REPRESENTADOS PELAS FEATURES QUE TEM E SUA CATEGORIA
# ({u':/': False, u'#LeaveEU': False, u'#ITV': False, u'#VoteLeave': False, u'#EU': 
# False, u'https': False, u'#UK': False, u'I': False, u'#BBC': False, u'#DavidCameron': False, 
# u'#Brexit': True, u'#StrongerIn': False, u'EU': False, u'The': False, u"Don't": False, 
# u'#EUref': True, u'#SKY': False, u'future': False, u'#BREXIT': False, u'UK': False, 
# u'#VoteRemain': True, u'If': False}, 'pos')
#
# IREMOS USA-LOS NO TREINAMENTO DO CLASSIFICADOR
######################################################################################


def avaliate_classifiers(featureSet):
	random.shuffle(featureSet)

	# Tem 2572 no FeatureSet, sendo: 1286 Stay e 1286 Leave
	training_set = featureSet[:2750]
	testing_set = featureSet[2750:]

	start_time = time.time()

	global classifier
	classifier = nltk.NaiveBayesClassifier.train(training_set)
	print("Naive Bayes Algo accuracy:", (nltk.classify.accuracy(classifier, testing_set)) * 100)
	classifier.show_most_informative_features(15)
	print("--- Classifier executed in %s seconds ---" % (time.time() - start_time))



########################################################################################

# PEGA OS TWEETS SALVOS NOS ARQUIVOS E TOKENIZA, TIRA STOPWORDS E REFERENCIAS NO TWITTER
filtered_tweets = reduce_tweets_words()

# CALCULA A DISTRIBUICAO DE FREQUENCIA DE TODAS AS PALAVRAS E DPS PEGA AS MAIS COMUNS
# ESSAS PALAVRAS MAIS COMUNS VAO SER AS FEATURES QUE USAREMOS PARA TREINAR O CLASSIFICADOR
top_tweets_features = getTop_tweet_words(filtered_tweets)

#Vai retornar uma tupla com o dict dizendo que features o documento tem => {u'even': True, u'story': False, ...}
# e que categoria esse dict de features representa
# return: ({u'even': True, u'story': False, ...}, neg)
#features_timer = time.time()

# filtered_tweets tem a tupla ([u'@mpvine', u'If', u'fifty', u'million', u'people', u'say',
# u'foolish', u'thing', u"it's", u'still', u'foolish', u'thing'], 'pos')) --> (tweet, category)
featureSet = [(find_features(tweet), category) for (tweet, category) in filtered_tweets]
#print(featureSet[2])
print(len(featureSet))

classifier = None
avaliate_classifiers(featureSet)

print('\n')
new_tweet = reduce_tweet("Sad to hear that some parents (but not most) are going to #VoteLeave leaving the childrens future to chance in #Timperley. #VoteRemain 1 retweet 0 likes ")
print(new_tweet)
new_tweet_feats = find_features(new_tweet)
print("Naive Bayes Result: ", (classifier.classify(new_tweet_feats)))
