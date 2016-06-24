from nltk.tokenize import TweetTokenizer
from TwitterAPI import TwitterAPI
from nltk.corpus import stopwords
# import ipdb
import time

"""
NLTK OBJs INIT
"""
tknzr = TweetTokenizer()
#tknzr.tokenize("STRING HERE")

"""
TWITTER SET UP
"""

#consumer
consumer_key = 'NZgTx9Oqe2ePnlwtPktgKLoCD'
consumer_secret = 'ZwrqJ5mgB5evpPm1jNNfpWPJ2DSlttFeIx3Ax2z0Rlk15uvc8g'

#token
access_token = 	'77996537-YzG4ydANL8xsmbKClpEb4pvV3BwMYgt0VZYREFy6Y'
access_token_secret = 'KlxURzE21r7HFPTjmiHpoXecIX7b0WfnyCyjekFnSRn34'

api = TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


"""
GET TWEETs
"""
#queries
query_stay = '#VoteStay OR #VoteRemain -#VoteLeave -RT'
query_leave = '#LeaveEu OR #VoteLeave -#VoteRemain -RT'

#tweets collected
documents_stay = []
documents_leave = []

#tweets interval controll
# - MY MAX : 99999999999999999999999999999
# - REAL ID: 743906289055580160
max_id = 99999999999999999999999999999

"""
collect 'stay' tweets
"""
stay_max_id = 99999999999999999999999999999
def get_stay_tweets():
	while len(documents_stay) < 500:
		stay_results = api.request('search/tweets', {'q':query_stay,'lang':'en','count':'100','until':'2016-06-10','max_id':str(stay_max_id),'include_entities':'false'})

		print(type(stay_results))

		#In fact, you can do much more with this syntax. The some_list[-n] syntax gets the nth-to-last element. 
		#So some_list[-1] gets the last element, some_list[-2] gets the second to last, etc, 
		global stay_max_id
		#stay_max_id = stay_results[stay_size -1]

		for item in stay_results:

			"""
			TOKENIZE ALL TWEETS
			"""
			tweet = item['text']
			#Sunstitui a quebra de linha por espaco
			tweet = tweet.replace('\n', ' ')
			#Pra replace muitos espacos em branco por um unico
			tweet = ' '.join(tweet.split())
			stay_max_id = item['id']
			documents_stay.append(tweet)
		print(len(documents_stay))


	print(len(documents_stay), 'Stay Tweets collecteds until now')

	print(documents_stay[3])
	with open('StayTweetsDate5.txt', 'w') as outfile:
		for item in documents_stay:
			#Pra ver se resolve o problema de
			#UnicodeEncodeError: 'ascii' codec can't encode character u'\u2026'
	  		outfile.write("%s\n" % item.encode('utf-8'))

leave_max_id = 99999999999999999999999999999
def get_leave_tweets():
	while len(documents_leave) < 500:
		leave_results = api.request('search/tweets', {'q':query_leave,'lang':'en','count':'100','until':'2016-06-10','max_id':str(leave_max_id),'include_entities':'false'})

		print(leave_results)

		global leave_max_id
		#leave_max_id = leave_results[-1]

		for item in leave_results:

			"""
			TOKENIZE ALL TWEETS
			"""
			tweet = item['text']
			#Sunstitui a quebra de linha por espaco
			tweet = tweet.replace('\n', ' ')
			#Pra replace muitos espacos em branco por um unico
			tweet = ' '.join(tweet.split())
			leave_max_id = item['id']
			documents_leave.append(tweet)
		print(leave_max_id)
		print(len(documents_leave))


	print(len(documents_leave), 'Stay Tweets collecteds until now')

	#print(documents_leave[3])
	with open('LeaveTweetsDate5.txt', 'w') as outfile:
		for item in documents_leave:
			#Pra ver se resolve o problema de
			#UnicodeEncodeError: 'ascii' codec can't encode character u'\u2026'
	  		outfile.write("%s\n" % item.encode('utf-8'))

get_leave_tweets()
get_stay_tweets()


# s0 = "Vote"
# s1 = "tp://tinyurl.com/of9l2py Why there's little hope for Greece's unemployed &amp; why they'll seek jobs in UK. #VoteLeave https://t.co/zUG8XBS9e8"
# s2 = "tp://tinyurl.com/of9l2py Why there's little hope for Greece's unemployed &amp; why they'll seek jobs in UK. #VoteLeave https://t.co/ZvAvbRYq43"

# def compare_similarity_tweets(new_tweet, old_tweet):
# 	len_tweet1 = len(new_tweet.split())
# 	len_tweet2 = len(old_tweet.split())

# 	words_in_common = 0
# 	for word in new_tweet.split():
# 		if word in old_tweet:
# 			words_in_common = words_in_common + 1

# 	print(words_in_common)
# 	print(len_tweet2)
# 	print(float(words_in_common)/float(len_tweet2))
# 	#Se as palavras do primeiro tweet ter quantidade maior do que 70% do segundo tweet
# 	#Entao provavelmente sao iguais e retornamos TRUE para nao adicionar o tweet1
# 	if float(words_in_common)/float(len_tweet2) > 0.7:
# 		return True
# 	else:
# 		return False


# if all(word in s2 for word in s1):
# 	print("Super parecidas")
# else:
# 	print("Nao parecem")

# if(compare_similarity_tweets(s1,s2)):
# 	print("Sao iguais porra!")
# else:
# 	print("Sao diferentes")