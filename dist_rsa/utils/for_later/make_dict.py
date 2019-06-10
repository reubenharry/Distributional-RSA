import pickle
from collections import defaultdict
import copy

#frequencies = pickle.load(open("data/ngrams_freqs","rb"))
#print(len(frequencies))
#print(frequencies)
frequencies = defaultdict(int)
#frequencies = pickle.load(open("data/ngrams_freqs","rb"))
for letter in list("abcdefghijklmnopqrstuvwxyz"):
#for letter in ['b']:
	f = open("data/google_unigrams/just_"+letter,'r')
	print(f)
	new_frequencies = defaultdict(int)
	for line in f:
#		print(line.split())
		if len(line.split())==4:
			key=line.split()[0].lower()
			value=int(line.split()[2])
			#print(value)
			#print(value > 10)
			if value > 1000:
				print(key,value)
				new_frequencies[key] += value

#	print("wrap")
#	for i in new_frequencies:
#		if new_frequencies[i] < 1000:
#			del new_frequencies[i]
	frequencies.update(new_frequencies)		
	f.close()
pickle.dump(frequencies,open("data/ngrams_freqs","wb"))
print(frequencies["big"])
print(frequencies["the"])


