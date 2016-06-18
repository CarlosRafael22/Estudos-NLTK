import nltk
import random
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier


from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI #inhirate from Classifier class
from statistics import mode
from nltk.tokenize import word_tokenize
## So pra contar o tempo gasto em cada execucao
import time
import pickle

################################################################
#
# ESCOLHENDO SE VAI ANALISAR OS MOVIE_REVIEWS OU SHORT_REVIEWS
#
#################################################################
documents = []
def movie_reviews_words():

	# Vou colocar todos os documentos 
	
	for category in movie_reviews.categories():
		for fileid in movie_reviews.fileids(category):
			documents.append((movie_reviews.words(fileid), category))

	#random.shuffle(documents)
	#Retorna desse jeito: ([u'plot', u':', u'a', u'human', u'space', u'astronaut', ...], u'pos')
	print(documents[0])
	print(documents[1800])
	# print('\n')

	# Vai ver quais sao as words mais frequentes em todos os documentos
	all_words = nltk.FreqDist(movie_reviews.words()) #retorna tuplas (words, frequency)
	return all_words


def short_reviews_words():

	short_pos = open("positive.txt", "r").read()
	short_neg = open("negative.txt", "r").read()
	#documents = []
	# for r in short_pos.readlines():
	# 	documents.append((r, "pos"))
	with open("positive.txt") as pos:
		lines = pos.readlines()
		for r in lines:
			documents.append((r, "pos"))

	# for r in short_neg.readlines():
	# 	documents.append((r, "neg"))
	with open("negative.txt") as neg:
		lines = neg.readlines()
		for r in lines:
			documents.append((r, "neg"))

	all_words = []
	#Tokenize todas as palavras de pos e neg
	# short_pos_words = word_tokenize(documents1)
	# short_neg_words = word_tokenize(documents2)

	# for w in short_pos_words:
	# 	all_words.append(w.lower())

	# for w in short_neg_words:
	# 	all_words.append(w.lower())
	print(documents[2])
	all_lines = [lines[0] for lines in documents]
	print(all_lines[1])
	#Vai pegar cada linha e tokenizar pra dps juntar todas em all_words
	
	# for line in all_lines:
	# 	all_words.append(word_tokenize(line))
	print(word_tokenize(all_lines[1]))
	# all_words = [word_tokenize(line) for line in all_lines]
	# print(all_words[1])

	#Vou transformar all_words em Set pq ai a ordem nao importa e 
	#tem como fazer o JOIN e deixar todas as palavras em um mesmo array
	all_words = set()
	for line in all_lines:
		line_tokenized = set(word_tokenize(line))
		all_words.union(line_tokenized)
	print(len(all_words))

	all_words = nltk.FreqDist(all_words)
	return all_words


#################################
# CHAMANDO AQUI
all_words = movie_reviews_words()

#all_words = short_reviews_words()


######################################################################


# Fazendo o stopwords nas palavras dos documentos pra tirar muita coisa inutil
stop_words = set(stopwords.words("english"))

	#Antes tava tirando as stopwords padrao do movie_reviews
	#Mas agora nao precisa executar isso ja que estou adicionando mais no stopwords
#all_words_no_stopwords = [w for w in movie_reviews.words() if w not in stop_words]
#all_words_no_stopwords = nltk.FreqDist(all_words_no_stopwords)
#print(all_words_no_stopwords.most_common(20))


#Vou tentar melhorar a lista de stopwords colocando nela algumas pontuacoes q nao servem de nada
# Fiz elas com unicode pq eh assim que as stop_words estao
punctuation = [u'.', u'-', u',', u'"', u'(', u')', u':', u'?', u"'", u'--', u';', u'!', u'$', u'*']
punctuation = set(punctuation)
new_stop_words = stop_words.union(punctuation)

#Pegando todas as palavras do movie_reviews sem as stopwords
stopwords_timer = time.time()
#all_words_no_stopwords = [w for w in movie_reviews.words() if w not in new_stop_words]
all_words_no_stopwords = [w for w in all_words if w not in new_stop_words]
all_words_no_stopwords = nltk.FreqDist(all_words_no_stopwords)
print("--- Stopwords executed in %s seconds ---" % (time.time() - stopwords_timer))
print('\n')
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
features_timer = time.time()
featureSet = [(find_features(rev), category) for (rev, category) in documents]
print("--- Find_features executed in %s seconds ---" % (time.time() - features_timer))
print('\n')

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


def scikit_classifiers(featureSet):
	#Vai testar em documentos positivos

	#Pra nao ter os neg e pos separados certinhos
	random.shuffle(featureSet)

	#Esse eh pro short_review databse
	# training_set = featureSet[:10000]
	# testing_set = featureSet[10000:]

	training_set = featureSet[:1900]
	testing_set = featureSet[1900:]

	#Vai testar em documentos negativos
	#Os 100 primeiros sao de teste e o resto de training
	#training_set = featureSet[100:]
	#testing_set = featureSet[:100]

	start_time = time.time()

	classifier = nltk.NaiveBayesClassifier.train(training_set)
	print("Naive Bayes Algo accuracy:", (nltk.classify.accuracy(classifier, testing_set)) * 100)
	#nb_time = time.time() - start_time
	print("--- Classifier executed in %s seconds ---" % (time.time() - start_time))
	
	MNB_classifier = SklearnClassifier(MultinomialNB())
	MNB_classifier.train(training_set)
	print("MNB_classifier accuracy:", (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100)
	#mnb_time = time.time() - nb_time
	print("--- MNB_classifier executed in %s seconds ---" % (time.time() - start_time))

	# GaussianNB_classifier = SklearnClassifier(GaussianNB())
	# GaussianNB_classifier.train(training_set)
	# print("GaussianNB_classifier accuracy:", (nltk.classify.accuracy(GaussianNB_classifier, testing_set)) * 100)
	
	BernoulliNB_classifier = SklearnClassifier(MultinomialNB())
	BernoulliNB_classifier.train(training_set)
	print("BernoulliNB_classifier accuracy:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100)
	#bnb_time = time.time() - mnb_time
	print("--- BernoulliNB_classifier executed in %s seconds ---" % (time.time() - start_time))
	

	#LogisticRegression, SGDClassifier
	#SVC, LinearSVC, NuSVC

	LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
	LogisticRegression_classifier.train(training_set)
	print("LogisticRegression_classifier accuracy:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)
	#bnb_time = time.time() - mnb_time
	print("--- Classifier executed in %s seconds ---" % (time.time() - start_time))

	SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
	SGDClassifier_classifier.train(training_set)
	print("SGDClassifier accuracy:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set)) * 100)
	#bnb_time = time.time() - mnb_time
	print("--- Classifier executed in %s seconds ---" % (time.time() - start_time))

	# SVC_classifier = SklearnClassifier(SVC())
	# SVC_classifier.train(training_set)
	# print("SVC accuracy:", (nltk.classify.accuracy(SVC_classifier, testing_set)) * 100)
	# #bnb_time = time.time() - mnb_time
	# print("--- Classifier executed in %s seconds ---" % (time.time() - start_time))

	LinearSVC_classifier = SklearnClassifier(LinearSVC())
	LinearSVC_classifier.train(training_set)
	print("LinearSVC accuracy:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)
	#bnb_time = time.time() - mnb_time
	print("--- Classifier executed in %s seconds ---" % (time.time() - start_time))

	NuSVC_classifier = SklearnClassifier(NuSVC())
	NuSVC_classifier.train(training_set)
	print("NuSVC accuracy:", (nltk.classify.accuracy(NuSVC_classifier, testing_set)) * 100)
	#bnb_time = time.time() - mnb_time
	print("--- Classifier executed in %s seconds ---" % (time.time() - start_time))

	voted_classifier = VoteClassifier(classifier, MNB_classifier, BernoulliNB_classifier,
						LogisticRegression_classifier, SGDClassifier_classifier, 
						LinearSVC_classifier, NuSVC_classifier)
	print("Voted_classifier accuracy:", (nltk.classify.accuracy(voted_classifier, testing_set)) * 100)
	
	print("Classification:", voted_classifier.classify(testing_set[0][0]), " Confidence %:", voted_classifier.confidence(testing_set[0][0]))
	
	print("Classification:", voted_classifier.classify(testing_set[1][0]), " Confidence %:", voted_classifier.confidence(testing_set[1][0]))
	
	print("Classification:", voted_classifier.classify(testing_set[2][0]), " Confidence %:", voted_classifier.confidence(testing_set[2][0]))
	
	print("Classification:", voted_classifier.classify(testing_set[3][0]), " Confidence %:", voted_classifier.confidence(testing_set[3][0]))
	
	print("Classification:", voted_classifier.classify(testing_set[4][0]), " Confidence %:", voted_classifier.confidence(testing_set[4][0]))
	



##########################################################
#
# Classe pra fazer a contagem dos votos de cada classifier e
# dai tirar qual eh a categoria que eh mais provavel que seja
#
###########################################################

class VoteClassifier(ClassifierI):
	#Construtor
	# Vamos passar uma lista de classifiers pra ele
	def __init__(self, *classifiers):
		self.classifiers = classifiers

	def classify(self, features):
		votes = []
		for c in self.classifiers:
			v = c.classify(features)
			votes.append(v)
		return mode(votes)

	def confidence(self, features):
		votes = []
		for c in self.classifiers:
			v = c.classify(features)
			votes.append(v)

		# Count how many times the popular votes has in the list
		choice_votes = votes.count(mode(votes))
		conf = float(choice_votes) / float(len(votes))
		return conf

########################################################
# Chamo um tipo de treinamento ou outro aqui
#########################################################

#Vai pedir pro usuario escolher entre o simple_training ou o cross_validation
# raw_input will parse any input as string
# Isso pq to usando no Python 2.7 do laptop
user_input = raw_input("Escolha como vai ser o treinamento:" + "\n" + "1) Simple_training" + "\n" + "2) Cross validation " +
						"\n" + "3) Scikitlearn classifiers" + "\n")
print(user_input)

#So pra ser inicializada e poder defini-la no ELIF para poder coloca-la no Pickle file junto com o classifier do cross validation
acc_list = None

#Convertendo pra int
while type(user_input) is not int:	
	try:
		user_input = int(user_input)
		
	except ValueError:
		#global user_input
		user_input = raw_input("Escolha como vai ser o treinamento:" + "\n" + "1) Simple_training" + "\n" + "2) Cross validation " +
						"\n" + "3) Scikitlearn classifiers" + "\n")
		print(user_input)
		#Return to the start of the loop
		# continue
	else:
		if user_input == 1:
			simple_training(featureSet)
		elif user_input == 2:
			global acc_list
			acc_list = _10_fold_cross_validation(featureSet)
			print(acc_list)
			print(calculate_average(acc_list))
		elif user_input == 3:
			scikit_classifiers(featureSet)
		break




##########################################################
# Vai usar Pickle para salvar os documentos e o classifier ja treinado
# Pq se for usar varios algoritmos pra classificar e tiver q treinar toda vez vai consumir muito tempo
# What pickle does is serialize, or de-serialize, python objects. This could be lists, dictionaries, or even things like our trained classifier!

#Salvando o classifier no arquivo pickle
print(type(classifier))
#print(acc_list)
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