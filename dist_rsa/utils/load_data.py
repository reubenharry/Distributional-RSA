from collections import Counter
import nltk
import pickle
import numpy as np
from nltk.corpus import wordnet as wn
from dist_rsa.utils.refine_vectors import h_dict
import csv
import scipy


controls = [["The man is a lion","brave","yellow","brave",'0'],
    ["The man is a lion","bold","furry","bold",'1'],
    ["The man is a lion","cowardly","yellow","cowardly",'2'],
    ["Love is a poison","destructive","liquid","destructive",'0`']
    ]

metaphors = [('brain','muscle'),('brain','computer'),('brain','giant'),('life','dream'),
    ('life','gift'),('life','party'),('book','hug'),('body','city'),('biking','gateway'),
    ('bed','heaven'),('athletics','drug'),('ascent','journey'),
    ('room','dungeon'),('room','furnace'),('room','matchbox'),('room','wok'),
    ('rose','woman'),('school','game'),('ship','palace'),('sin','sickness'),('skin','parchment'),
    ('market','graveyard'),('sun','hammer'),('photographer','fly'),('place','junkyard'),('pornography','cancer'),
    ('principal','dictator'),('pub','family'),('life','train'),('life','snowflake'),('life','river'),('mother','mosquito'),
    ('music','language'),('relationship','treasure'),('voice','river'),('tongue','razor')]

twitter_vecs = pickle.load(open("dist_rsa/data/word_vectors/glove.twitter.27B.mean_vecs25",'rb'))

metaphors = [x for x in metaphors if x[0] in twitter_vecs and x[1] in twitter_vecs]
metaphors = sorted(metaphors,key=lambda x:scipy.spatial.distance.cosine(twitter_vecs[x[0]],twitter_vecs[x[1]]) ,reverse=True)

sentences = ["The principal is a dictator","The room is a dungeon","Biking is a gateway","The room is a furnace",
"Skin is a parchment","The place is a junkyard","The market is a graveyard","Pornography is a cancer",
"The room is a wok","Sin is a sickness","The room is a matchbox",
"The ascent is a journey","Athletics is a drug","The mother is a mosquito","The photographer is a fly"]

pickle.dump({'shark':np.array([1.0,1.0]),'human':np.array([0.0,0.0]),
	'fish':np.array([1.0,0.0]),'swims':np.array([1.0,0.0]),'predator':np.array([0.0,1.0])},
	open("dist_rsa/data/word_vectors/"+"handmadeplain2",'wb'))

animals = ['fish', 'sheep', 'ant', 'monkey', 'bear', 'bee', 'goose', 'dog', 'horse', 'puppy', 'owl', 'frog', 'dolphin', 'tiger', 'bird', 'wolf', 'cat', 'duck', 'pig', 'rabbit', 'kangaroo', 'lion', 'penguin', 'goat', 'elephant', 'buffalo', 'ox', 'whale', 'zebra', 'human', 'bat', 'fox', 'shark', 'cow']
animal_features = []
imported_adjs = "ant small strong busy large weak idle goose loud mean annoying quiet nice agreeable bat scary blind nocturnal unalarming sighted diurnal horse fast strong beautiful slow weak ugly bear scary big fierce unalarming small nonviolent kangaroo jumpy bouncy cute relaxed inelastic unattractive bee busy small angry idle large unangry lion ferocious scary strong nonviolent unalarming weak bird free graceful small unfree awkward large monkey funny smart playful humorless stupid unplayful buffalo big strong wild small weak tame owl wise quiet nocturnal foolish loud diurnal cat independent lazy soft dependent fast hard ox strong big slow weak small fast cow fat dumb lazy thin smart fast penguin cold cute funny hot unattractive humorless dog loyal friendly happy disloyal unfriendly unhappy pig dirty fat smelly clean thin fragrant dolphin smart friendly playful stupid unfriendly unplayful rabbit fast furry cute slow hairless unattractive duck loud cute quacking quiet unattractive non-quacking shark scary dangerous mean unalarming safe nice elephant huge smart heavy small stupid light sheep wooly fluffy dumb hairless hard smart fish scaly wet smelly smooth dry fragrant tiger striped fierce scary unpatterned nonviolent unalarming fox sly smart pretty artless stupid ugly whale large graceful majestic small awkward inferior frog slimy noisy jumpy nonslippery quiet relaxed wolf scary mean angry unalarming nice unangry goat funny hungry loud humorless full quiet zebra striped exotic fast unpatterned native slow".split()
for i in range(len(imported_adjs)):
	if i%7 != 0:
		animal_features.append(imported_adjs[i])
animal_features = list(set(animal_features))


def load_vecs(mean,pca,vec_length,vec_type):
	name=vec_type+h_dict[(pca,mean)]+str(vec_length)
	message = "\nloading vecs (load vecs): "+name
	print(message)
	vecs = pickle.load(open("dist_rsa/data/word_vectors/"+name,'rb'))
	return vecs
#improved words
def get_words():
	nouns = {}
	adjs = {}
	words = set()

	vecs = pickle.load(open("dist_rsa/data/word_vectors/glove.6B.mean_vecs300",'rb'),encoding='latin1')

	with open('dist_rsa/data/concreteness.csv', newline='') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		for i,row in enumerate(spamreader):
			if i>0:
				is_bigram = float(row[1])!=0
				is_noun = (row[8])=='Noun'		
				is_adj = (row[8])=='Adjective'
				is_adv = (row[8])=='Adverb'
				if not is_bigram:
					if is_noun:
						nouns[row[0]]=float(row[2])
					if is_adj:
						adjs[row[0]]=float(row[2])

		return nouns,adjs




def freq_count(text):
	counts = Counter()
	for word in text:
		counts[word] += 1
	return counts

def compute_frequencies():
	freqs = pickle.load(open("dist_rsa/data/ngrams_freqs","rb"))
	return freqs

def get_freqs(preprocess=True):

	if not preprocess:
		freqs = pickle.load(open("dist_rsa/data/google_freqs",'rb'))
	else:
		freqs = compute_frequencies()
		pickle.dump(freqs,open("dist_rsa/data/google_freqs","wb"))
	return freqs

# frequencies,nouns,adjectives,verbs = get_words(preprocess=True)

# nouns,adjectives = sorted(nouns, key=lambda x: frequencies[x], reverse=True), sorted(adjectives, key=lambda x: frequencies[x], reverse=True)

def initialize(words,max_number=1000000,load_pickle=False,file_input='coca',save_path=""):


	if load_pickle:
		text = open(file_input,'r').read().lower()
		text = nltk.word_tokenize(text)

		print("NUMBER OF WORDS IN TEXT: " + str(len(text)))

		frequencies = freq_count(text)
		frequencies_list = sorted(list(frequencies.items()),key = lambda x : x[1],reverse=True)[:max_number]
		top_words = [x[0] for x in frequencies_list]

		print("Top Words: ", top_words[:5])

		text = [x for x in text if x in top_words]
		text = list(set(text))


		#intersperse, to trick the stupid ngrams in the nltk pos tagger
		new_text = []
		for i,x in enumerate(text):
			new_text.append(x)
			new_text.append('.')
		text = new_text
		text = dict(nltk.pos_tag(text))


		for word in words:
			if word[0] not in text:
				# print(word[0] + ' was not in text: now added')
				text[word[0]] = word[1]
				frequencies[word[0]] += 1


		nouns = [x for x in text if text[x] in ["NN",'NNP']]
		adjectives = [x for x in text if text[x] in ["JJ"]]

		pickle.dump(frequencies,open(save_path+"frequency_count","wb"))
		pickle.dump(nouns,open(save_path+"nouns","wb"))
		pickle.dump(adjectives,open(save_path+"adjectives","wb"))


	else:
		frequencies = pickle.load(open(save_path+"frequency_count",'rb'))
		nouns = pickle.load(open(save_path+"nouns",'rb'))
		adjectives = pickle.load(open(save_path+"adjectives",'rb'))






	return frequencies,nouns,adjectives
