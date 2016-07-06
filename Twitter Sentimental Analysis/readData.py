import ast
import numpy as np

def readData():

	idStorage = []

	stay = []
	leave =[]
	other = []

	"""

	ADA's FILE

	"""
	with open('1tweets_leave_stay_eu_2016-06-22.csv','r') as file:
		first = True
		for tweet_tkns in file:
			if first:
				first = False
			else:
				splited = tweet_tkns.split(';')
				id = splited[0]
				tkns = ast.literal_eval(splited[1])
				tkns = [token.lower() for token in tkns]
				tkns = [token.decode('utf-8') for token in tkns]
				target = splited[2]

				if id not in idStorage:
					idStorage.append(id)

					if str.startswith(target,'leave'):
						leave.append(tkns)
					elif str.startswith(target,'stay'):
						stay.append(tkns)
					elif str.startswith(target,'other'):
						other.append(tkns)
					else:
						raise("CLASSE INEXISTENTE!!!")
				else:
					print('ID DUPLICADO E IGNORADO: ', id)

	"""
	CARLINHOS's FILES
	"""

	from nltk.tokenize import TweetTokenizer
	tknzr = TweetTokenizer()

	with open('1stayTweets.txt','r') as file:
		for tweet in file:

			tweet_tokens = tknzr.tokenize(tweet)
			#tweet_tokens = [token.encode('utf-8').decode('utf-8') for token in tweet_tokens]

			#DEixando tudo em unicode
			#Pra tirar se tiver emotions no formato /u2026 por exemplo
			# tweet_tokens = [token.decode('unicode_escape').encode('utf-8','ignore') for token in tweet_tokens]
			
			tweet_tokens = [w.lower() for w in tweet_tokens]
			tweet_tokens = [token.encode('utf-8').decode('utf-8') for token in tweet_tokens]
			stay.append(tweet_tokens)

	with open('1leaveTweets.txt','r') as file:
		for tweet in file:

			tweet_tokens = tknzr.tokenize(tweet)
			#tweet_tokens = [token.encode('utf-8').decode('utf-8') for token in tweet_tokens]

			#DEixando tudo em unicode
			#Pra tirar se tiver emotions no formato /u2026 por exemplo
			# tweet_tokens = [token.decode('unicode_escape').encode('utf-8','ignore') for token in tweet_tokens]
			# tweet_tokens = [token.decode('utf-8') for token in tweet_tokens]
			tweet_tokens = [w.lower() for w in tweet_tokens]
			tweet_tokens = [token.encode('utf-8').decode('utf-8') for token in tweet_tokens]

			leave.append(tweet_tokens)

	with open('1noneTweets.txt','r') as file:
		for tweet in file:

			tweet_tokens = tknzr.tokenize(tweet)
			#tweet_tokens = [token.encode('utf-8').decode('utf-8') for token in tweet_tokens]

			#DEixando tudo em unicode
			#Pra tirar se tiver emotions no formato /u2026 por exemplo
			# tweet_tokens = [token.decode('unicode_escape').encode('utf-8','ignore') for token in tweet_tokens]
			# tweet_tokens = [token.decode('utf-8') for token in tweet_tokens]
			tweet_tokens = [w.lower() for w in tweet_tokens]
			tweet_tokens = [token.encode('utf-8').decode('utf-8') for token in tweet_tokens]
			other.append(tweet_tokens)

	return [leave, stay, other]