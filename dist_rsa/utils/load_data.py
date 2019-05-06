from collections import Counter
import nltk
import pickle
import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn
from dist_rsa.utils.refine_vectors import h_dict
import csv
import scipy


# controls = [["The man is a lion","brave","yellow","brave",'0'],
#     ["The man is a lion","bold","furry","bold",'1'],
#     ["The man is a lion","cowardly","yellow","cowardly",'2'],
#     ["Love is a poison","destructive","liquid","destructive",'0`']
#     ]


control_set = {
	("love","poison"): ["noxious","harmful","potent","dangerous","cruel","strange","insidious","mysterious","destructive","vicious"],
	("principal","dictator"): ["authoritarian", "autocratic", "beloved", "brutal", "corrupt", "cruel", "notorious", "oppresive", "paranoid", "tyrannical"],
	('room','dungeon') : ["airy","bearable","beautiful","claustrophobic","creepy","ghoulish","hellish","macabre","nightmarish","primeval",],
	('sin','sickness') : ["despicable","detestable","earthly","evil","horrible","incurable","survivable","wicked","widespread","worsening"],
	('athletics','drug') : ["addictive","corrupt","dangerous","effective","lucrative","notorious","rampant","serious","widespread","promising"],
	('voice','river') : ["authentic","continuous","dominant","dynamic","eloquent","forceful","lyrical","natural","soulful","wild"],
	('life','river') : ["beautiful","broad","ancient","extensive","majestic","navigable","precious","romantic","treacherous","wild",],
	('life','dream') : ["crazy","elusive","epic","fabulous","ideal","magical","memorable","normal","strange","wonderful",],
				}

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

# twitter_vecs = pickle.load(open("dist_rsa/data/word_vectors/glove.twitter.27B.mean_vecs25",'rb'))

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

	# vecs = pickle.load(open("dist_rsa/data/word_vectors/glove.6B.mean_vecs300",'rb'),encoding='latin1')

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

def get_possible_utterances_and_quds(subj,pred,word_selection_vecs):

	from dist_rsa.utils.config import abstract_threshold,concrete_threshold
	
	nouns,adjs = get_words(with_freqs=False)
	# print('abstract_threshold',abstract_threshold)
	# print('concrete_threshold',concrete_threshold)
	noun_words = [n for n in nouns if nouns[n] > concrete_threshold and n in word_selection_vecs]
	possible_utterances = sorted(noun_words,\
		key=lambda x: scipy.spatial.distance.cosine(word_selection_vecs[x],np.mean([word_selection_vecs[subj],word_selection_vecs[subj]],axis=0)),reverse=False)
		# key=lambda x:freqs[x],reverse=True)

	qud_words = [a for a in list(adjs) if adjs[a] < abstract_threshold and a in word_selection_vecs]
	quds_near_subj = sorted(qud_words,\
		key=lambda x:scipy.spatial.distance.cosine(word_selection_vecs[x],word_selection_vecs[subj]),reverse=False)

	quds_near_pred = sorted(qud_words,\
		key=lambda x:scipy.spatial.distance.cosine(word_selection_vecs[x],word_selection_vecs[pred]),reverse=False)
		# key=lambda x:freqs[x],reverse=True)

	return possible_utterances,[val for pair in zip(quds_near_subj, quds_near_pred) for val in pair]

def load_trofi_data():
	data = iter(open("dist_rsa/experiment/trofi_data.txt",'r').readlines())
	
	results = {}
	literal_results = []
	non_literal_results = []
	
	word = None

	enter_literals = True
	i = 0
	while True:
		# print(i)
		i += 1
		try:
			line = next(data).strip('\n')
			
		except StopIteration:
			results[word] = {
				'literal': literal_results,
				'non_literal': non_literal_results,
			}
			break
			
		if line == '********************':
			pass # do nowt
		elif line.startswith('***'):
			# dump results so far
			results[word] = {
				'literal': literal_results,
				'non_literal': non_literal_results,
			}

			word = line.lstrip('***').rstrip('***')
			literal_results = []
			non_literal_results = []

		elif line == '*literal cluster*':
			enter_literals = True
			pass # start loading literals
		elif line == '*nonliteral cluster*':
			enter_literals = False
			pass # start loading nonliterals
		elif line.startswith('w'):
			val,line =  line.split("\t")[1],line.split("\t")[2]

			if enter_literals and val=="L":
				literal_results.append(line)
			elif not enter_literals and val=="N":
				non_literal_results.append(line)



	# data = data.split("********************")
	# data = [section.split("*literal cluster*") for section in data]

	del results[None]
	return results

def load_tsvetkov_data():

	xls = pd.ExcelFile("dist_rsa/data/TsvetkovEtAl_ACL2014_testsets.xlsx")

	lit_dict = pd.read_excel(xls,"LIT_SVO_EN")
	met_dict = pd.read_excel(xls,"MET_SVO_EN.txt")

	lits = []
	mets = []

	for i in range(110):

		lits.append({"verb":lit_dict["verb"][i],"nsubj":lit_dict["subject"][i],"dobj":lit_dict["object"][i],"sent":lit_dict["sentence"][i]})
		mets.append({"verb":met_dict["verb"][i],"nsubj":met_dict["subject"][i],"dobj":met_dict["object"][i],"sent":met_dict["sentence"][i]})

	return lits,mets


