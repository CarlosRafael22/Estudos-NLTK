# coding: utf-8
import nltk
import random
from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier


from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI #inhirate from Classifier class
#from statistics import mode
## So pra contar o tempo gasto em cada execucao
import time
import pickle

import re
from nltk.tokenize import TweetTokenizer, word_tokenize

import collections
from nltk.metrics import precision, recall, f_measure
from readData import readData

from confusion import confusion


import string
from nltk import bigrams
########################################################

tknzr = TweetTokenizer()
new_stop_words = set()
tokenized_tweets = []



#Vao ser usados para guardar os tweets pos e neg em separado pra quando for olhar um tweet
#que sera usado pra treinamento a gnt ve se ele ja existe ou nao
#Assim nao armazena tweets iguais e classificar melhor
positive_tweets = []
negative_tweets = []

def openFile_getTokenizedTweets(filename, category):
	with open(filename) as doc:
		lines = doc.readlines()
		#print(len(lines))
		for l in lines:
			#Pra tirar se tiver emotions no formato /u2026 por exemplo
			l = l.decode('unicode_escape').encode('ascii','ignore')
			tokens = tknzr.tokenize(l)
			global tokenized_tweets
			#Pega cada token e bota em minuscula
			lw_tokens = [w.lower() for w in tokens]
			tokenized_tweets.append((lw_tokens, category))

def getTokenizedTweetsFile(filename, category):
	with open(filename) as doc:
		lines = doc.readlines()
		result = []
		#print(len(lines))
		for l in lines:
			#Pra tirar se tiver emotions no formato /u2026 por exemplo
			#l = l.decode('unicode_escape').encode('ascii','ignore')
			l = l.decode('unicode_escape').encode('utf-8','ignore')
			#l = l.encode('utf-8')
			tokens = tknzr.tokenize(l)
			#Pega cada token e bota em minuscula
			lw_tokens = [w.lower() for w in tokens]
			result.append((lw_tokens, category))
	return result

def categorizy_tweets(tweets, category):
	tweets_cat = [(tweet, category) for tweet in tweets]
	return tweets_cat

def reduce_tweets_words():

	[leave_tweets, stay_tweets, other_tweets] = readData()

	leave_tweets = categorizy_tweets(leave_tweets, "neg")
	new_leave = getTokenizedTweetsFile("leaveTweets/ExtraLeaveTweets.txt", "neg")
	leave_Farias = getTokenizedTweetsFile("leaveTweets/FariasLeave.txt", "neg")
	stay_tweets = categorizy_tweets(stay_tweets, "pos")
	new_stay = getTokenizedTweetsFile("stayTweets/ExtraStayTweets.txt", "pos")
	stay_Farias = getTokenizedTweetsFile("stayTweets/FariasStay.txt", "pos")
	other_tweets = categorizy_tweets(other_tweets, "neutral") 

	tokenized_tweets = leave_tweets + new_leave + leave_Farias + stay_tweets + new_stay + stay_Farias + other_tweets + other_tweets
	all_words = []

	print(len(leave_tweets))
	print(len(new_leave))
	print(len(leave_Farias))
	print(len(stay_tweets))
	print(len(new_stay))
	print(len(stay_Farias))
	print(len(other_tweets))
	print(len(other_tweets))
	print(len(tokenized_tweets))
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
	punctuation = [u'.', u'-', u',', u'"', u'(', u')', u':', u"'", u'--', u';', 
	u'!', u'$', u'*', u'&', u'...', u':/', u'/', u'..']
	punctuation = set(punctuation)

	punct = list(string.punctuation)
	#stop = stopwords.words('english') + punctuation + ['rt', 'via']
	global new_stop_words
	new_stop_words = stop_words.union(punct)

	twitter_symbols = [u'rt', u'#voteleave', u'#voteremain', u'#leaveeu', u'h', u'#rt', u'=', u'@', u'https',
	u'+', u'\'', u'|', u'...']
	twitter_symbols = set(twitter_symbols)
	new_stop_words = new_stop_words.union(twitter_symbols)

	# NA VERDADE NAO TO CONSEGUINDO TIRAR O @USER DO RT MAS ISSO
	# NAO VAI INTERFERIR POIS A FREQUENCIA DE SE TER UM @USER DO MESMO USER EH POUCA
	#user_rt_pattern = "@\w+?"
	#url_pattern = 'http[s]:/'
	emotions_pattern = '\u\d+'
	url_pattern = 'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'
	user_rt_pattern = '(?:@[\w_]+)'
	# "(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
 #    r'(?:[\w_]+)', # other words
 #    r'(?:\S)' # anything else
    #user_rt_pattern = '(?:@[\w_]+)'
	
	

	# filtered_tweets = [for tweet_cat in tokenized_tweets if ]
	filtered_tweets = []
	tokens_to_be_removed = []
	#print(tokenized_tweets[228:230])

	#Para cada tupla (tweet_tok,categoria)
	for tweet_cat in tokenized_tweets:

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

		#Limpar o tokens_to_be_removed pq senao vai sempre acumular de outros tweets
		tokens_to_be_removed = []
		#Adiciona o tweet sem as stopwords na nova lista
		filtered_tweets.append(tweet_cat)

	# Exemplo de tweet filtrado com stopwords
	# ([u'@mpvine', u'If', u'fifty', u'million', u'people', u'say',
	# u'foolish', u'thing', u"it's", u'still', u'foolish', u'thing'], 'pos')
	#print(filtered_tweets[228:230])


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
	#print(text_tokenized)

	emotions_pattern = '\u\d+'
	url_pattern = 'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'
	user_rt_pattern = '(?:@[\w_]+)'

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
  	# print(all_tweets_words[1230:1240])
  	print('\n')
  	all_tweets_words = nltk.FreqDist(all_tweets_words)
  	#print(all_tweets_words["#itv"])
  	print(all_tweets_words.most_common(65))
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

def avaliate_new_classifier(featureSet):
	print("Vamos treinar o classificador agora!")
	print("\n")
	#random.shuffle(featureSet)

	#Cada um tem 198 + 50
	positive_tweets = featureSet[:247]

	#Misturando as paradas pra nao ficar testando só os mesmos últimos
	random.shuffle(positive_tweets)

	#print(featureSet[7185])
	#Pra pegar 7185 do pos e 7185 do negativo mas o negativo tem 7213
	negative_tweets = featureSet[247:345]
	random.shuffle(negative_tweets)

	neutral_tweets = featureSet[345:]
	random.shuffle(neutral_tweets)

	#Agora vou dividir cada classe em um conjunto de referencia e outro de teste
	pos_cutoff = len(positive_tweets)*3/4
	neg_cutoff = len(negative_tweets)*3/4
	neu_cutoff = len(neutral_tweets)*3/4

	# 75% dos tweets vao pra ser de referencia(treinamento) e o resto pra teste
	pos_references = positive_tweets[:pos_cutoff]
	pos_tests = positive_tweets[pos_cutoff:]

	neg_references = negative_tweets[:neg_cutoff]
	neg_tests = negative_tweets[neg_cutoff:]

	neu_references = neutral_tweets[:neu_cutoff]
	neu_tests = neutral_tweets[neu_cutoff:]

	#COnjunto de treinamento e de testes pra calcular a accuracy
	training_set = pos_references + neg_references + neu_references
	testing_set = pos_tests + neg_tests + neu_tests

	start_time = time.time()

	global classifier
	print("Comecou a treina-lo agora!")

	#training_set2 = [(t,l) for (t,l,twe) in training_set]

	classifier = nltk.NaiveBayesClassifier.train(training_set)
	#classifier = SklearnClassifier(LinearSVC())
	#classifier.train(training_set)

	#testing_set2 = [(t,l) for (t,l,twe) in testing_set]
	print("Naive Bayes Algo accuracy:", (nltk.classify.accuracy(classifier, testing_set)) * 100)
	classifier.show_most_informative_features(60)

	refsets = collections.defaultdict(set)
	testsets = collections.defaultdict(set)

	for i, (feats, label) in enumerate(testing_set):
	    refsets[label].add(i)
	    observed = classifier.classify(feats)
	    testsets[observed].add(i)
	 
	print 'pos precision:', precision(refsets['pos'], testsets['pos'])
	print 'pos recall:', recall(refsets['pos'], testsets['pos'])
	print 'pos F-measure:', f_measure(refsets['pos'], testsets['pos'])

	print 'neg precision:', precision(refsets['neg'], testsets['neg'])
	print 'neg recall:', recall(refsets['neg'], testsets['neg'])
	print 'neg F-measure:', f_measure(refsets['neg'], testsets['neg'])

	print 'neutral precision:', precision(refsets['neutral'], testsets['neutral'])
	print 'neutral recall:', recall(refsets['neutral'], testsets['neutral'])
	print 'neutral F-measure:', f_measure(refsets['neutral'], testsets['neutral'])

	conf = confusion(refsets, testsets)
	print(conf)
	print("--- Classifier executed in %s seconds ---" % (time.time() - start_time))



######################################################################################
#
# CRIANDO O METODO PRA FAZER O 10-FOLD CROSS-VALIDATION E TENTAR MELHORAR OS RESULTADOS
#
######################################################################################

#Retorna a lista das accuracies dos 10 treinamentos
def _10_fold_cross_validation(featureSet):
	start_time = time.time()

	#Agora vou tentar criar um 10-fold training set pra ver se tem melhor desempenho
	num_folds = 10
	subset_size = len(featureSet)/num_folds
	# Como no total tem 14.400 tweets coletados, cada subset tera 1.440 tweets
	random.shuffle(featureSet)

	accuracy_list = []
	for i in range(num_folds):
		testing_this_round = featureSet[i*subset_size:][:subset_size]
		training_this_round = featureSet[:i*subset_size] + featureSet[(i+1)*subset_size:]

		print("Round "+ str(i) +" : ")
		print("Testing fold is: " + "featureSet[" +str(i*subset_size)+":"+str((i+1)*subset_size)+"]")
		#print(len(testing_this_round))
		#print(len(training_this_round))
		global classifier
		classifier = nltk.NaiveBayesClassifier.train(training_this_round)
		accuracy_list.append(((nltk.classify.accuracy(classifier, testing_this_round)) * 100))
		print("--- Classifier executed in %s seconds ---" % (time.time() - start_time))
	return accuracy_list

#Calcula a media dos acurracies do 10-fold
def calculate_average(list):
		acc_total = 0
		for i in range(len(list)):
			acc_total = acc_total + list[i]
		return acc_total/len(list)

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
avaliate_new_classifier(featureSet)
# acc_list = _10_fold_cross_validation(featureSet)
# print(acc_list)
# print(calculate_average(acc_list))

print('\n')
#new_tweet = reduce_tweet("Sad to hear that some parents (but not most) are going to #VoteLeave leaving the childrens future to chance in #Timperley. #VoteRemain 1 retweet 0 likes ")
#new_tweet = reduce_tweet("Brexit is basically a referendum on how white people in the UK feel about minorities: http://bit.ly/28SyaDW ")

#new_tweet = reduce_tweet("#voteleave Wednesday is my 25th wedding anniversary, when we vote out I'll buy you brexiters all a drink to cap a momentous week.")
#new_tweet = reduce_tweet("Many of my father's generation gave their lives in defence of #Europe I won't give up their prize lightly! #VoteRemain")

#new_tweet = reduce_tweet("I love how the leave campaigns only argument is that they will prevent immigration... Yet they won't even be able to do that. #VoteRemain")
# new_tweet = reduce_tweet("John Oliver on Brexit: 'Britain would be absolutely crazy to leave' the EU https://t.co/PXiDId3yXi #VoteRemain #StrongerIn #euref")
# print(new_tweet)
# new_tweet_feats = find_features(new_tweet)
# print("Naive Bayes Result: ", (classifier.classify(new_tweet_feats)))
