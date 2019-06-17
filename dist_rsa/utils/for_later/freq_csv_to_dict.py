import csv
import pickle
from collections import defaultdict
from dist_rsa.utils.load_data import load_vecs

vecs = load_vecs(mean=False,pca=False,vec_length=300,vec_type='glove.6B.')

freqs = defaultdict(int)

with open('dist_rsa/data/google_freqs/unigrams', 'r') as csvfile:
	spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
	for row in spamreader:
		if row[0] in vecs:
			print(row)
			freqs[row[0]]+=int(row[-2])


pickle.dump(freqs,open('dist_rsa/data/google_freqs/freqs','wb'))