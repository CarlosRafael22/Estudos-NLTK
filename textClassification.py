import nltk
import random
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords

## So pra contar o tempo gasto em cada execucao
import time

import pickle

documents = []
for category in movie_reviews.categories():
	for fileid in movie_reviews.fileids(category):
		documents.append((movie_reviews.words(fileid), category))

random.shuffle(documents)

# print(documents[0])
# print('\n')

# Vai ver quais sao as words mais frequentes em todos os documentos
all_words = nltk.FreqDist(movie_reviews.words()) #retorna tuplas (words, frequency)

# Fazendo o stopwords nas palavras dos documentos pra tirar muita coisa inutil
stop_words = set(stopwords.words("english"))

all_words_no_stopwords = [w for w in movie_reviews.words() if w not in stop_words]
all_words_no_stopwords = nltk.FreqDist(all_words_no_stopwords)
#print(all_words_no_stopwords.most_common(20))


#Vou tentar melhorar a lista de stopwords colocando nela algumas pontuacoes q nao servem de nada
# Fiz elas com unicode pq eh assim que as stop_words estao
punctuation = [u'.', u'-', u',', u'"', u'(', u')', u':', u'?', u"'", u'--', u';', u'!', u'$', u'*']
punctuation = set(punctuation)
new_stop_words = stop_words.union(punctuation)

#Pegando todas as palavras do movie_reviews sem as stopwords
all_words_no_stopwords = [w for w in movie_reviews.words() if w not in new_stop_words]
all_words_no_stopwords = nltk.FreqDist(all_words_no_stopwords)
print(all_words_no_stopwords.most_common(20))

print(len(all_words_no_stopwords)) #Tem 39608 palavras
#print(all_words_no_stopwords.most_common()[25000]) #Pega a palavra 25000
#Pra saber o numero de vezes que certa palavra apareceu
print(all_words_no_stopwords["stupid"])

#Pega as primeiras 25000 palavras da tupla pra servir como os features
# que vao ser usadas como parametros pra avaliar positivo ou negativo
#word_features = list(all_words_no_stopwords.keys())[:50]

#Pega as 20.000 palavras mais comuns de todos os reviews para servirem de features ao avaliarmos novos reviews
top_word_features = all_words_no_stopwords.most_common(3000) #retorna (u'revolutionaries', 3)

#Como top_wf retorna (word,freq) iremos pegar so as palavras que sao as keys
top_word_features_keys = [wf[0] for wf in top_word_features]
#print(top_word_features_keys[550:575])

#Retorna uma lista com True ou False dizendo quais palavras da word_features o documento tem
# retorna: {u'even': True, u'story': False, u'also': True, u'see': True, u'much': False,.... }
def find_features(document):
	# Pega todas as palavras do documento e transforma em set pra retornar as palavras independente da frequencia dela
	words = set(document)
	features = {}
	counter = 0
	#print(top_word_features_keys[:20])
	for w in top_word_features_keys:
		features[w] = (w in words)

	return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

#Vai retornar uma tupla com o dict dizendo que features o documento tem => {u'even': True, u'story': False, ...}
# e que categoria esse dict de features representa
# return: ({u'even': True, u'story': False, ...}, neg)
featureSet = [(find_features(rev), category) for (rev, category) in documents]


###########################################################################################
# Agora que pegamos as tuplas com a representacao das features e categorias podemos treinar o algoritmo
###########################################################################################



##################################################################################################
# TEM DOIS JEITOS DE CLASSIFICAR E MEDIR A ACCURACY:
# 1)
# PEGANDO SIMPLESMENTE OS 1900 PRIMEIROS MOVIE_REVIEWS PARA TREINAREM O CLASSIFIER 
# E OS 100 ULTIMOS MOVIE_REVIEWS PARA SEREM OS QUE SERAO TESTADOS E MEDIR ACCURACY EM CIMA DISSO
# 2)
# DIVIDINDO OS MOVIE_REVIEWS EM 10-FOLDS E FICAR VARIANDO O CONJUNTO DE TESTES DO PRIMEIRO FOLD ATE O ULTIMO
# CALCULANDO A ACCURACY EM CADA ITERACAO E DPS CALCULA A MEDIA DE TODAS AS ACCURACIES
##################################################################################################

classifier = None
def simple_training(featureSet):
	start_time = time.time()
	#Tem 2000 movie_reviews entao 100 ficam para serem testados
	# print(featureSet[1])
	training_set = featureSet[:1900]
	testing_set = featureSet[1900:]

	#Usaremos o Naive Bayes pra treinar e testar
	# Naive Bayes: posterior prob = prior occurence x likelihood / evidence

	# Dizendo que eu to pegando o classifier global pq senao ele so vai usar a variavel localmente
	# global classifier
	#classifier = nltk.NaiveBayesClassifier.train(training_set)
	global classifier
	try:
		classifier_f = open("naiveBayes.pickle", "rb")
		print(type(classifier_f))
		classifier = pickle.load(classifier_f)
		print(type(classifier))
		classifier_f.close()
	except IOError:
		print("Nao tem pickle ainda, vai usar o classifier")
		#global classifier
		classifier = nltk.NaiveBayesClassifier.train(training_set)

	print("Naive Bayes Algo accuracy:", (nltk.classify.accuracy(classifier, testing_set)) * 100)
	classifier.show_most_informative_features(15)
	print("--- Classifier executed in %s seconds ---" % (time.time() - start_time))


def _10_fold_cross_validation(featureSet):
	start_time = time.time()
	#Agora vou tentar criar um 10-fold training set pra ver se tem melhor desempenho
	num_folds = 10
	subset_size = len(featureSet)/num_folds
	accuracy_list = []

	global classifier
	try:
		classifier_f = open("naiveBayesCrossValidation.pickle", "rb")
		print(type(classifier_f))
		classifier = pickle.load(classifier_f)
		#Pega o segundo obj que foi serializado, no caso eh a lista de accuracy
		accuracy_list = pickle.load(classifier_f)
		print(type(accuracy_list))
		classifier_f.close()
	except IOError:
		print("Nao tem pickle ainda pra cross validation, vai usar o classifier")
		# global classifier
		# classifier = nltk.NaiveBayesClassifier.train(training_set)
		for i in range(num_folds):
			testing_this_round = featureSet[i*subset_size:][:subset_size]
			training_this_round = featureSet[:i*subset_size] + featureSet[(i+1)*subset_size:]

			print("Round "+ str(i) +" : ")
			print("Testing fold is: " + "featureSet[" +str(i*subset_size)+":"+str((i+1)*subset_size)+"]")
			#print(len(testing_this_round))
			#print(len(training_this_round))
			classifier = nltk.NaiveBayesClassifier.train(training_this_round)
			accuracy_list.append(((nltk.classify.accuracy(classifier, testing_this_round)) * 100))
			print("--- Classifier executed in %s seconds ---" % (time.time() - start_time))
	print("--- Classifier executed in %s seconds ---" % (time.time() - start_time))
	return accuracy_list


def calculate_average(list):
		acc_total = 0
		for i in range(len(list)):
			acc_total = acc_total + list[i]
		return acc_total/len(list)


########################################################
# Chamo um tipo de treinamento ou outro aqui
#########################################################

#Vai pedir pro usuario escolher entre o simple_training ou o cross_validation
user_input = input("Escolha como vai ser o treinamento:" + "\n" + "1) Simple_training" + "\n" + "2) Cross validation ")
print(user_input)

#So pra ser inicializada e poder defini-la no ELIF para poder coloca-la no Pickle file junto com o classifier do cross validation
acc_list = None

if user_input == 1:
	simple_training(featureSet)
elif user_input == 2:
	global acc_list
	acc_list = _10_fold_cross_validation(featureSet)
	#print(acc_list)
	#print(calculate_average(acc_list))




##########################################################
# Vai usar Pickle para salvar os documentos e o classifier ja treinado
# Pq se for usar varios algoritmos pra classificar e tiver q treinar toda vez vai consumir muito tempo
# What pickle does is serialize, or de-serialize, python objects. This could be lists, dictionaries, or even things like our trained classifier!

#Salvando o classifier no arquivo pickle
print(type(classifier))
print(acc_list)
if(user_input == 1):
	save_classifier = open("naiveBayes.pickle", "wb")
	if classifier is not None:
		pickle.dump(classifier, save_classifier)
		save_classifier.close()
	else:
		print("Classifier foi None e nao criou pickle file")
elif(user_input == 2):
	save_classifier = open("naiveBayesCrossValidation.pickle", "wb")
	if classifier is not None:
		pickle.dump(classifier, save_classifier)
		pickle.dump(acc_list, save_classifier)
		save_classifier.close()
	else:
		print("Classifier foi None e nao criou pickle file cross_validation")

# movie_reviews.words(fileid) volta todas as palavras com unicode, ou seja, u'plot' ao inves de 'plot'
# com esse metodo abaixo a gnt pega cada palavra dentro do movie_reviews.words(fileid)
# tira o unicode e so vai retornar a palavra. Mas deixa o processo mais lento

'''
documents = []
for category in movie_reviews.categories():
	for fileid in movie_reviews.fileids(category):
		fileidWords = []
		for w in movie_reviews.words(fileid):
			fileidWords.append(str(w))
		documents.append((fileidWords, category))
'''

# NO tutorial ele fez isso mas todas ja vem em minuscula
# all_words = []
# for w in movie_reviews.words():
# 	all_words.append(w.lower())


# print(all_words)
'''
documents = [ list((movie_reviews.words(fileid), category)
				for category in movie_reviews.categories()
				for fileid in movie_reviews.fileids(category)) ]
'''