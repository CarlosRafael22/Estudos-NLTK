from TwitterAPI import TwitterAPI

"""
TWITTER SET UP
"""

class TwitterWrap:

	def __init__(self):
		#consumer
		consumer_key = 'NZgTx9Oqe2ePnlwtPktgKLoCD'
		consumer_secret = 'ZwrqJ5mgB5evpPm1jNNfpWPJ2DSlttFeIx3Ax2z0Rlk15uvc8g'

		#token
		access_token = 	'77996537-YzG4ydANL8xsmbKClpEb4pvV3BwMYgt0VZYREFy6Y'
		access_token_secret = 'KlxURzE21r7HFPTjmiHpoXecIX7b0WfnyCyjekFnSRn34'

		self.api = TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

	def query(self,query_str,count=100, max_id=9999999999999999999999):
		return self.api.request('search/tweets', {'q':query_str,'lang':'en','count':str(count),'max_id':str(max_id),'include_entities':'false'})