import json
import pickle
from collections import defaultdict

a = json.loads(open("/Users/reuben/Documents/COCA/COCA-ngrams/coca-an-2grams.json",'r').read())


adjnouns = open("training_data/AN_metaphor.txt",'r')

freqs = defaultdict(int)

for x in list(adjnouns):

	x=x[:-1]
	print(x)

	adj,noun = tuple(x.split(" "))

	# try:

	keys1 = [x for x in list(a) if x.split('_')[0]==adj]
	# list_at_key1 = a[key1]
	# print(list_at_key1)

	for key1 in keys1:

		keys2 = [x for x in list(a[key1]) if x.split('_')[0]==noun]

		for key2 in keys2:

			print(key1,"key1")
			print(key2,"key2")

			freqs[(adj,noun)]+=a[key1][key2]



		# freqs[(adj,noun)]=a[key1][key2]
		# print(a[key1][key2])

	# except:
	# 	freqs[(adj,noun)]=0

	# print()


pickle.dump(freqs,open("frequencies2",'wb'))

# print(freqs)

# # print(a[key1][key2])

# frequencies = open("frequencies",'w')

# print(json.dumps(freqs))

# frequencies.write(json.dumps(freqs))

# print(a[""])