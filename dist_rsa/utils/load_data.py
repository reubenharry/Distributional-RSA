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

test_metaphors = {("pain","thorn"):"Pain is a thorn", 
	("life","train"):"Life is a train", 
	("life","tree"):"Life is a tree", 
	("life","snowflake"):"Life is a snowflake", 
	("music","language"):"Music is a language",  
	("restaurant","circus"):"The restaurant is a circus", 
	("voice","hammer"):"The voice is a hammer", 
	("voice","stone"):"The voice is a stone", 
	("man","snake"):"The man is a snake",
	("father","shark"):"The father is a shark.",
	("curiosity","disease"):"Curiosity is a disease",
	("art","woman"):"Art is a woman",
	("earth","womb"):"The earth is a womb",
	("home","lifeboat"):"Home is a lifeboat",
	("hope","drug"):"Hope is a drug"
	}

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
	if vec_type=="word2vec":
		message = "\nloading vecs (load vecs): word2vec"
		print(message)
		from gensim.models.keyedvectors import KeyedVectors
		vecs = pickle.load(open('dist_rsa/data/word_vectors/word2vec_pickle','rb'))
		if mean or pca: raise Exception("no mean or pca version available")
		return vecs 
	name=vec_type+h_dict[(pca,mean)]+str(vec_length)
	message = "\nloading vecs (load vecs): "+name
	print(message)
	vecs = pickle.load(open("dist_rsa/data/word_vectors/"+name,'rb'))
	return vecs
#improved words
def get_words(with_freqs=False):
	nouns = {}
	adjs = {}
	words = set()

	vecs = pickle.load(open("dist_rsa/data/word_vectors/glove.6B.mean_vecs300",'rb'),encoding='latin1')

	with open('dist_rsa/data/concreteness.csv', newline='') as csvfile:
		r = csv.reader(csvfile, delimiter=',', quotechar='|')
		for i,row in enumerate(r):
			if i>0:
				is_bigram = float(row[1])!=0
				is_noun = (row[8])=='Noun'		
				is_adj = (row[8])=='Adjective'
				is_adv = (row[8])=='Adverb'
				freq = row[7]
				if not is_bigram:
					if is_noun:
						if with_freqs:nouns[row[0]]=float(row[2]),freq
						else: nouns[row[0]]=float(row[2])
					if is_adj:
						if with_freqs: adjs[row[0]]=float(row[2]),freq
						else: adjs[row[0]]=float(row[2])

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

