import ast
import numpy as np
import ipdb

def readData():

	idStorage = []

	stay = []
	leave =[]
	other = []

	"""

	ADA's FILE

	"""
	with open('tweets_leave_stay_eu_2016-06-22.csv','r',encoding='utf-8') as file:
		first = True
		for tweet_tkns in file:
			if first:
				first = False
			else:
				splited = tweet_tkns.split(';')
				id = splited[0]
				tkns = ast.literal_eval(splited[1])
				tkns = tkns.lower()
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

	with open('stayTweets.txt','r',encoding='utf-8') as file:
		for tweet in file:

			tweet_tokens = tknzr.tokenize(tweet)
			tweet_tokens = [token.encode('utf-8').decode('utf-8') for token in tweet_tokens]

			tweet_tokens = [w.lower() for w in tweet_tokens]
			stay.append(tweet_tokens)

	with open('leaveTweets.txt','r',encoding='utf-8') as file:
		for tweet in file:

			tweet_tokens = tknzr.tokenize(tweet)
			tweet_tokens = [token.encode('utf-8').decode('utf-8') for token in tweet_tokens]

			tweet_tokens = [w.lower() for w in tweet_tokens]

			leave.append(tweet_tokens)

	with open('noneTweets.txt','r',encoding='utf-8') as file:
		for tweet in file:

			tweet_tokens = tknzr.tokenize(tweet)
			tweet_tokens = [token.encode('utf-8').decode('utf-8') for token in tweet_tokens]

			tweet_tokens = [w.lower() for w in tweet_tokens]
			other.append(tweet_tokens)

	return [leave, stay, other]