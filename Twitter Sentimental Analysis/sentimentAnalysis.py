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

import collections
from nltk.metrics import precision, recall, f_measure
from readData import readData
########################################################

tknzr = TweetTokenizer()
new_stop_words = set()
tokenized_tweets = []



#Vao ser usados para guardar os tweets pos e neg em separado pra quando for olhar um tweet
#que sera usado pra treinamento a gnt ve se ele ja existe ou nao
#Assim nao armazena tweets iguais e classificar melhor
positive_tweets = []
negative_tweets = []

#Se retornar TRUE quer dizer que o new_tweet eh muito parecido com o old_tweet, ou seja,
#eh o mesmo tweet só com algumas palavras diferentes, como links.
def compare_similarity_of_tweets(new_tweet, old_tweet):
	len_tweet1 = len(new_tweet.split())
	len_tweet2 = len(old_tweet.split())

	words_in_common = 0
	for word in new_tweet.split():
		if word in old_tweet:
			words_in_common = words_in_common + 1

	# print(words_in_common)
	# print(len_tweet2)
	# print(float(words_in_common)/float(len_tweet2))
	#Se as palavras do primeiro tweet ter quantidade maior do que 70% do segundo tweet
	#Entao provavelmente sao iguais e retornamos TRUE para nao adicionar o tweet1
	if len_tweet2 == 0:
		return True
	if float(words_in_common)/float(len_tweet2) > 0.7:
		return True
	else:
		return False


# def openFile_getTokenizedTweets(filename, category):
# 	with open(filename) as doc:
# 		start_time = time.time()

# 		lines = doc.readlines()
# 		print(len(lines))
# 		lw_tokens = []
# 		for l in lines:
# 			#Pra tirar se tiver emotions no formato /u2026 por exemplo
# 			l = l.decode('unicode_escape').encode('ascii','ignore')

# 			#Antes de tokenizar os tweets eu vou ver se esse tweet ja existe para nao coloca-lo
# 			# de novo e causar erro na classificao

# 			if category == "pos":
# 				tweet_already_exists = False
# 				#Pra cada tweet vai ver se ele ja existe na lista ou nao
# 				if len(positive_tweets) == 0:
# 					#So vou botar o texto do tweet nessa lista
# 					positive_tweets.append(l)
# 				else:					
# 					for tweet in positive_tweets:
# 						if compare_similarity_of_tweets(l, tweet):
# 							tweet_already_exists = True
# 							break

# 					#Se esse tweet nao existe ainda entao vamos tokeniza-lo e adicionar em tokenized_tweets
# 					if tweet_already_exists == False:
# 						tokens = tknzr.tokenize(l)
# 						#Pega cada token e bota em minuscula
# 						lw_tokens = [w.lower() for w in tokens]
# 						positive_tweets.append(l)
# 			elif category == "neg":
# 				tweet_already_exists = False
# 				#Pra cada
# 				if len(negative_tweets) == 0:
# 					negative_tweets.append(l)
# 				else:					
# 					for tweet in negative_tweets:
# 						if compare_similarity_of_tweets(l, tweet):
# 							tweet_already_exists = True
# 							break

# 					#Se esse tweet nao existe ainda entao vamos tokeniza-lo e adicionar em tokenized_tweets
# 					if tweet_already_exists == False:
# 						tokens = tknzr.tokenize(l)
# 						#Pega cada token e bota em minuscula
# 						lw_tokens = [w.lower() for w in tokens]
# 						negative_tweets.append(l)

# 			global tokenized_tweets
# 			tokenized_tweets.append((lw_tokens, category))
# 	print(len(tokenized_tweets))
# 	print("--- Read file and tokenized tweets in %s seconds ---" % (time.time() - start_time))
					


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
			l = l.decode('unicode_escape').encode('ascii','ignore')
			tokens = tknzr.tokenize(l)
			#Pega cada token e bota em minuscula
			lw_tokens = [w.lower() for w in tokens]
			result.append((lw_tokens, category))
	return result

def categorizy_tweets(tweets, category):
	tweets_cat = [(tweet, category) for tweet in tweets]
	return tweets_cat

def reduce_tweets_words():

	#Ja vai pegar os tweets dos txts e tokeniza-los e colocar nesse array como tuplas
	# (Twitter_tokenizado , categoria_twitter)
	# ([u'RT', u'@mpvine', u':', u'If', u'fifty', u'million', u'people', u'say', u'a', 
	# u'foolish', u'thing', u',', u"it's", u'still', u'a', u'foolish', u'thing', u'.'], 'pos')
	

	# Tem 2853 no FeatureSet, sendo: 1286 Stay e 1567 Leave
	# openFile_getTokenizedTweets("StayTweets1.txt", "pos")
	# openFile_getTokenizedTweets("StayTweetsDate.txt", "pos")
	# openFile_getTokenizedTweets("StayTweetsDate2.txt", "pos")
	# openFile_getTokenizedTweets("StayJune14.txt", "pos")
	# openFile_getTokenizedTweets("StayJune15.txt", "pos")
	# openFile_getTokenizedTweets("StayJune16.txt", "pos")
	# openFile_getTokenizedTweets("StayJune17.txt", "pos")
	# openFile_getTokenizedTweets("StayJune18.txt", "pos")
	# openFile_getTokenizedTweets("StayJune19.txt", "pos")
	# openFile_getTokenizedTweets("StayJune20.txt", "pos")
	# openFile_getTokenizedTweets("StayTweetsNow.txt", "pos")

	
	###########################################################################################

	#Too fazendo isso pra pegar so os 1286 primeiros desse arquivo q tem 1537
	# with open("LeaveTweets1.txt") as doc:
	# 	lines = doc.readlines()
	# 	lines = lines[:1286]
	# 	#print(len(lines))
	# 	for l in lines:
	# 		#Pra tirar se tiver emotions no formato /u2026 por exemplo
	# 		l = l.decode('unicode_escape').encode('ascii','ignore')
	# 		tokens = tknzr.tokenize(l)
	# 		global tokenized_tweets
	# 		#Pega cada token e bota em minuscula
	# 		lw_tokens = [w.lower() for w in tokens]
	# 		tokenized_tweets.append((lw_tokens, "neg", l))

	# openFile_getTokenizedTweets("LeaveTweetsDate.txt", "neg")
	# openFile_getTokenizedTweets("LeaveTweetsDate2.txt", "neg")
	# openFile_getTokenizedTweets("LeaveJune14.txt", "neg")
	# openFile_getTokenizedTweets("LeaveJune15.txt", "neg")
	# openFile_getTokenizedTweets("LeaveJune16.txt", "neg")
	# openFile_getTokenizedTweets("LeaveJune17.txt", "neg")
	# openFile_getTokenizedTweets("LeaveJune18.txt", "neg")
	# openFile_getTokenizedTweets("LeaveJune19.txt", "neg")
	# openFile_getTokenizedTweets("LeaveJune20.txt", "neg")
	# openFile_getTokenizedTweets("LeaveTweetsNow.txt", "neg")

	[leave_tweets, stay_tweets, other_tweets] = readData()

	leave_tweets = categorizy_tweets(leave_tweets, "neg")
	new_leave = getTokenizedTweetsFile("ExtraLeaveTweets.txt", "neg")
	stay_tweets = categorizy_tweets(stay_tweets, "pos")
	new_stay = getTokenizedTweetsFile("ExtraStayTweets.txt", "pos")
	other_tweets = categorizy_tweets(other_tweets, "neutral") 

	tokenized_tweets = leave_tweets + new_leave + stay_tweets + new_stay + other_tweets
	all_words = []

	print(len(leave_tweets))
	print(len(new_leave))
	print(len(stay_tweets))
	print(len(new_stay))
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
	punctuation = [u'.', u'-', u',', u'"', u'(', u')', u':', u'?', u"'", u'--', u';', 
	u'!', u'$', u'*', u'&', u'...', u':/', u'/', u'%', u'..']
	punctuation = set(punctuation)
	global new_stop_words
	new_stop_words = stop_words.union(punctuation)

	twitter_symbols = [u'rt', u'#voteleave', u'#voteremain', u'#leaveeu', u'h', u'#rt', u'=', u'@', u'https']
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


def avaliate_classifiers(featureSet):
	print("Vamos treinar o classificador agora!")
	print("\n")
	#random.shuffle(featureSet)

	#Vai fazer o calculo de recall e precision
	# You need to build 2 sets for each classification label:
	# a reference set of correct values, and a test set of observed values.

	#Os primeiros 6686 + 500(dia 14) tweets sao positivos e resto(6757 + 500(dia 14)) negativo
	positive_tweets = featureSet[:7185]

	#Misturando as paradas pra nao ficar testando só os mesmos últimos
	random.shuffle(positive_tweets)

	#print(featureSet[7185])
	#Pra pegar 7185 do pos e 7185 do negativo mas o negativo tem 7213
	negative_tweets = featureSet[7185:14372]
	random.shuffle(negative_tweets)

	#Agora vou dividir cada classe em um conjunto de referencia e outro de teste
	pos_cutoff = len(positive_tweets)*3/4
	neg_cutoff = len(negative_tweets)*3/4

	# 75% dos tweets vao pra ser de referencia(treinamento) e o resto pra teste
	pos_references = positive_tweets[:pos_cutoff]
	pos_tests = positive_tweets[pos_cutoff:]

	neg_references = negative_tweets[:neg_cutoff]
	neg_tests = negative_tweets[neg_cutoff:]

	#COnjunto de treinamento e de testes pra calcular a accuracy
	training_set = pos_references + neg_references
	testing_set = pos_tests + neg_tests

	start_time = time.time()

	global classifier
	print("Comecou a treina-lo agora!")

	#training_set2 = [(t,l) for (t,l,twe) in training_set]

	classifier = nltk.NaiveBayesClassifier.train(training_set)
	#testing_set2 = [(t,l) for (t,l,twe) in testing_set]
	print("Naive Bayes Algo accuracy:", (nltk.classify.accuracy(classifier, testing_set)) * 100)
	classifier.show_most_informative_features(30)

	refsets = collections.defaultdict(set)
	testsets = collections.defaultdict(set)

	# for i, (feats, label, l) in enumerate(testing_set):
	#     refsets[label].add(i)
	#     observed = classifier.classify(feats)
	#     testsets[observed].add(i)
	#     print("--"*200)
	#     print()
	#     print("Classified as: ",observed)
	#     print()
	#     print(l)
	#     print()
	#     print("--"*200)
	#     raw_input("Press any key to continue:")
	 
	print 'pos precision:', precision(refsets['pos'], testsets['pos'])
	print 'pos recall:', recall(refsets['pos'], testsets['pos'])
	print 'pos F-measure:', f_measure(refsets['pos'], testsets['pos'])
	print 'neg precision:', precision(refsets['neg'], testsets['neg'])
	print 'neg recall:', recall(refsets['neg'], testsets['neg'])
	print 'neg F-measure:', f_measure(refsets['neg'], testsets['neg'])


	print("--- Classifier executed in %s seconds ---" % (time.time() - start_time))

def avaliate_new_classifier(featureSet):
	print("Vamos treinar o classificador agora!")
	print("\n")
	#random.shuffle(featureSet)

	#Cada um tem 197
	positive_tweets = featureSet[:196]

	#Misturando as paradas pra nao ficar testando só os mesmos últimos
	random.shuffle(positive_tweets)

	#print(featureSet[7185])
	#Pra pegar 7185 do pos e 7185 do negativo mas o negativo tem 7213
	negative_tweets = featureSet[196:293]
	random.shuffle(negative_tweets)

	neutral_tweets = featureSet[293:]
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
	#testing_set2 = [(t,l) for (t,l,twe) in testing_set]
	print("Naive Bayes Algo accuracy:", (nltk.classify.accuracy(classifier, testing_set)) * 100)
	classifier.show_most_informative_features(30)

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
new_tweet = reduce_tweet("Brexit is basically a referendum on how white people in the UK feel about minorities: http://bit.ly/28SyaDW ")
print(new_tweet)
new_tweet_feats = find_features(new_tweet)
print("Naive Bayes Result: ", (classifier.classify(new_tweet_feats)))
