def confusion(refsets,testsets):
	#true values
	pos_pos = 0
	neg_neg = 0
	neu_neu = 0

	#wrong pos label
	pos_neg = 0
	pos_neu = 0

	#wrong neg label
	neg_pos = 0
	neg_neu = 0

	#wrong neu label
	neu_pos = 0
	neu_neg	= 0

	for ref in testsets['pos']:

		if ref in refsets['pos']:
			pos_pos += 1
		elif ref in	refsets['neg']:
			pos_neg +=1
		elif ref in refsets['neutral']:
			pos_neu +=1

	for ref in testsets['neg']:

		if ref in refsets['neg']:
			neg_neg += 1
		elif ref in	refsets['pos']:
			neg_pos +=1
		elif ref in refsets['neutral']:
			neg_neu +=1

	for ref in testsets['neutral']:

		if ref in refsets['neutral']:
			neu_neu += 1
		elif ref in	refsets['pos']:
			neu_pos +=1
		elif ref in refsets['neg']:
			neu_neg +=1	

	return [
		[pos_pos, pos_neg, pos_neu],
		[neg_pos, neg_neg, neg_neu],
		[neu_pos, neu_neg, neu_neu]
	]