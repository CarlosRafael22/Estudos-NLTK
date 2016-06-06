import nltk #Se nao botar isso ele da erro 'global name nltk is not defined' qd for executar o codigo do Part of Speech
from nltk.corpus import state_union #So pra pegar um conjunto de textos de la
from nltk.tokenize import PunktSentenceTokenizer

############ PART OF SPEECH TAGGING

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

#Pra usar o Punkt ele tem que ser treinado pq eh um unsupervised tokenizer
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)

#print(tokenized)

def process_content():
	# i dentro do for vai ser cada sentenca. ENtao em cada uma a gnt tokenize por word
	# e dps faz o part of speech tagging de cada word e printa a frase tageada
	try:
		for i in tokenized:
			words = nltk.word_tokenize(i)
			tagged = nltk.pos_tag(words)
			print(tagged)

	except Exception as e:
		print(str(e))

process_content()
##################################################################

'''
POS tag list:

CC	coordinating conjunction
CD	cardinal digit
DT	determiner
EX	existential there (like: "there is" ... think of it like "there exists")
FW	foreign word
IN	preposition/subordinating conjunction
JJ	adjective	'big'
JJR	adjective, comparative	'bigger'
JJS	adjective, superlative	'biggest'
LS	list marker	1)
MD	modal	could, will
NN	noun, singular 'desk'
NNS	noun plural	'desks'
NNP	proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	predeterminer	'all the kids'
POS	possessive ending	parent's
PRP	personal pronoun	I, he, she
PRP$	possessive pronoun	my, his, hers
RB	adverb	very, silently,
RBR	adverb, comparative	better
RBS	adverb, superlative	best
RP	particle	give up
TO	to	go 'to' the store.
UH	interjection	errrrrrrrm
VB	verb, base form	take
VBD	verb, past tense	took
VBG	verb, gerund/present participle	taking
VBN	verb, past participle	taken
VBP	verb, sing. present, non-3d	take
VBZ	verb, 3rd person sing. present	takes
WDT	wh-determiner	which
WP	wh-pronoun	who, what
WP$	possessive wh-pronoun	whose
WRB	wh-abverb	where, when
'''