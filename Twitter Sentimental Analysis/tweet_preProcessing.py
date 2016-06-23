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
query_stay = '#VoteStay OR #VoteRemain -#VoteLeave'
query_leave = '#LeaveEu OR #VoteLeave -#VoteRemain'

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
while len(documents_stay) < 1000:

	stay_results = api.request('search/tweets', {'q':query_stay,'lang':'en','count':'100','max_id':str(max_id),'include_entities':'false'})
	
	for item in stay_results:

		#discover min e max ids got in the results query
		if len(documents_stay) == 0:
			max_id = item['id']

		else:

			if item['id'] < max_id:
				max_id = item['id']
				# print('since_id',since_id)

		"""
		TOKENIZE ALL TWEETS
		"""
		tweet = item['text']
		#Sunstitui a quebra de linha por espaco
		tweet = tweet.replace('\n', ' ')
		#Pra replace muitos espacos em branco por um unico
		tweet = ' '.join(tweet.split())
		documents_stay.append(tweet)


	print(len(documents_stay), 'Stay Tweets collecteds until now')
	print('Max:',max_id,)
	print('Sleeping')

	#update interval tweets control
	max_id -= 1

	time.sleep(3)

print(documents_stay[35])
with open('StayTweetsNow.txt', 'w') as outfile:
	for item in documents_stay:
		#Pra ver se resolve o problema de
		#UnicodeEncodeError: 'ascii' codec can't encode character u'\u2026'
  		outfile.write("%s\n" % item.encode('utf-8'))

"""
collect 'leave' tweets
"""

max_id = 99999999999999999999999999999

#collect 'stay' tweets
while len(documents_leave) < 1000:

	leave_results = api.request('search/tweets', {'q':query_leave,'lang':'en','count':'100','max_id':str(max_id),'include_entities':'false'})
	
	for item in leave_results:

		#discover min e max ids got in the results query
		if len(documents_leave) == 0:
			max_id = item['id']

		else:

			if item['id'] < max_id:
				max_id = item['id']
				# print('since_id',since_id)

		"""
		TOKENIZE ALL TWEETS
		"""
		tweet = item['text']
		#Sunstitui a quebra de linha por espaco
		tweet = tweet.replace('\n', ' ')
		#Pra replace muitos espacos em branco por um unico
		tweet = ' '.join(tweet.split())
		documents_leave.append(tweet)


	print(len(documents_leave), 'Leave Tweets collecteds until now')
	print('Max:',max_id,)
	print('Sleeping')

	#update interval tweets control
	max_id -= 1

	time.sleep(3)

#print(documents_leave[35])
with open('LeaveTweetsNow.txt', 'w') as outfile:
	for item in documents_leave:
  		outfile.write("%s\n" % item.encode('utf-8'))
# stop_words = set(stopwords.words("english"))

#print(documents_stay[35])
