def softmax(z):
	z = np.array(z)
	z.astype(float)
	return np.exp(z) / np.sum(np.exp(z))

def freq_count(sents):
	counts = Counter()
	import copy
	for sent in sents:
		for word in sent:
			counts[word] += 1
	dist = zip(*counts.items())
	return zip(dist[0],softmax(dist[1]))


# #takes a stream of sentences and concatenates any words which are listed in bigram
def munge(f):
	a = open(f,'r+')
	b = a.read()

	b = re.sub('brick wall','brickwall',b)
	a.write(b)
	a.close()