from twitterwrap import TwitterWrap
from nltk.tokenize import TweetTokenizer
import csv

tw = TwitterWrap()
tknzr = TweetTokenizer()

#tweets collected
documents_stay = []
documents_leave = []
documents_other = []

#tweets interval controll
# - MY MAX : 99999999999999999999999999999
# - REAL ID: 743906289055580160
max_id = 99999999999999999999999999999

set_size = 100

"""
collect classifyied tweets
"""
with open('tweets_leave_stay_eu.csv', 'w',encoding='utf-8', newline='') as csvfile:

	spamwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

	spamwriter.writerow(['id','tweet_tokens','target'])

	while len(documents_stay) < set_size or len(documents_leave) < set_size or len(documents_other) < set_size:

		docs = tw.query('(leave OR stay) AND EU')
		
		for item in docs:

			#discover min e max ids got in the results query
			if len(documents_stay) == 0:
				max_id = item['id']

			else:

				if item['id'] < max_id:
					max_id = item['id']
					# print('since_id',since_id)

			def fix(t,rt):
				t = t[:-3]
				size = 1
				index = 0
				while size < len(t) and size < len(rt):

					if t[len(t)-size:] != rt[:size]:
						size += 1
					else:
						if size > index:
							index = size
						size += 1

				return t+rt[index:]

			"""
			TOKENIZE ALL TWEETS
			"""

			if 'retweeted_status' in item:
				tweet = fix(item['text'], item['retweeted_status']['text'])
			else:
				tweet = item['text']

			tweet_tokens = tknzr.tokenize(tweet)
			tweet_tokens = [token.encode('utf-8').decode('utf-8') for token in tweet_tokens]


			ok = False
			while not ok:

				print()
				print('TWEET:',tweet)
				print('_'*150)
				print()	

				answer = input('Wich class? Type: ["S" or "s" for STAY] | ["L" or "l" for LEAVE]  | ["O" or "o" for OTHER]: ')
				answer.lower()

				print()

				if answer == 's':
					if len(documents_stay) < set_size:
						documents_stay.append(tweet_tokens)
						spamwriter.writerow([item['id'], tweet_tokens,'stay'])
					else:
						print('TWEET IGNORED - MAX SIZE REACHED FOR <STAY> DATA SET')

					ok = True
				elif answer == 'l':
					if len(documents_leave) < set_size:
						documents_leave.append(tweet_tokens)
						spamwriter.writerow([item['id'], tweet_tokens,'leave'])
					else:
						print('TWEET IGNORED - MAX SIZE REACHED FOR <LEAVE> DATA SET')
					ok = True
				elif answer == 'o':
					if len(documents_other) < set_size:
						documents_other.append(tweet_tokens)
						spamwriter.writerow([item['id'], tweet_tokens,'other'])
					else:
						print('TWEET IGNORED - MAX SIZE REACHED FOR <OTHER> DATA SET')
					ok = True
				else:
					print('Wrong input:',answer,"put 'S' or 'L' or 'O'")

				print()

				print(len(documents_stay), 'STAY Tweets collecteds until now')
				print(len(documents_leave), 'LEAVE Tweets collecteds until now')
				print(len(documents_other), 'OTHERs Tweets collecteds until now')
				print('Max:',max_id,)
				print()

				if len(documents_leave) == set_size and len(documents_stay) == set_size and len(documents_other) == set_size:
					print()
					print('DONE 1!')
					print()
					break;

			if len(documents_leave) == set_size and len(documents_stay) == set_size and len(documents_other) == set_size:
				print()
				print('DONE 2!')
				print()
				break;
    	