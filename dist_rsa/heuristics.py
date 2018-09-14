import nltk
import scipy
import numpy as np
import pickle
import itertools
from dist_rsa.utils.load_data import *
import spacy
nlp = spacy.load('en')
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
# from dist_rsa.utils.helperfunctions import *
# from dist_rsa.utils.config import abstract_threshold,concrete_threshold

# find non-stop words
# find matrix verb
# project glove vectors into subspace
# sentence = "The man waded through the paperwork."
vec_size,vec_kind = 300,'glove.6B.'
vecs = load_vecs(mean=True,pca=False,vec_length=vec_size,vec_type=vec_kind) 
# projection_matrix = np.asarray([vecs["man"],vecs["paperwork"]]).T
# print(scipy.spatial.distance.cosine(vecs["waded"],vecs["man"]+vecs["paperwork"]))
# projected_man = np.dot(np.expand_dims(vecs["man"],0), projection_matrix)
# projected_waded = np.dot(np.expand_dims(vecs["waded"],0), projection_matrix)
# projected_paperwork = np.dot(np.expand_dims(vecs["paperwork"],0), projection_matrix)
# print(scipy.spatial.distance.cosine(projected_waded,projected_man+projected_paperwork))
# from nltk.tokenize import RegexpTokenizer
# tokenizer = RegexpTokenizer(r'\w+')

def remove_stops(sent):
	without_stops = [str(w).lower() for w in sent if ((str(w) not in stops) and (len(str(w))>2) )]
	return without_stops

def sent_to_dict(sent,verb_lemma):



	parsed = nlp(sent)
	verb = [word for word in parsed if word.lemma_==verb_lemma][0]
	dobj = [word for word in verb.children if word.dep_=="dobj"]
	nsubj = [word for word in verb.children if (word.dep_=="nsubj" or word.dep_=="nsubjpass")]
	if remove_stops(dobj) == [] and remove_stops(nsubj) == []:
		print("BLAH BLAH")
		print(list([c.dep_=="nsubjpass" for c in verb.children]))
		# print(verb.children)
		raise Exception
	if not dobj==[]:
		dobj = list(dobj[0].subtree)
	if not nsubj==[]:
		nsubj = list(nsubj[0].subtree)
	out = {"sent":remove_stops(parsed),"dobj":remove_stops(dobj),"verb":verb_lemma,"nsubj":remove_stops(nsubj)}
	print(out)
	return out



def metrics(sent,verb_lemma):

	# sent = sent_to_dict(sent,verb_lemma)
	try:
		sent = sent_to_dict(sent,verb_lemma)
		print(sent)
	except: 
		print(sent)
		print("FAILURE")
		return None,None

	# print("begin sent",sent,"end sent")
	# print(sent.keys())

	projection_matrix = np.asarray([vecs[w] for w in sent["sent"] if w in vecs]).T

	# print(s.shape,projection_matrix.shape)
	# raise Exception
	s_o = np.mean([vecs[v] for v in sent["dobj"]+sent["nsubj"] if v in vecs],axis=0)
	# print("O",o)
	v = vecs[sent["verb"]]
	# print(o)
	# print(v)
	# s_o = np.mean([s,o],axis=0)

	distance_in_original_space = scipy.spatial.distance.cosine(v,s_o)


	projected_s_o = np.dot(np.expand_dims(s_o,0), projection_matrix)
	projected_v = np.dot(np.expand_dims(v,0), projection_matrix)

	distance_in_subspace = scipy.spatial.distance.cosine(projected_s_o,projected_v)

	return distance_in_original_space, distance_in_original_space-distance_in_subspace

data = load_trofi_data()

# print(data["absorb"]["literal"])
# keys = list(data.keys())
# print(keys,"keys")


lit_data = list(zip(*[metrics(s,"absorb") for s in data["absorb"]["literal"]+data["assault"]["literal"]  ]))
met_data = list(zip(*[metrics(s,"absorb") for s in data["absorb"]["non_literal"]+data["assault"]["non_literal"]  ]))

# print(lit_data)

# print(metrics(data["absorb"]["literal"][0],"absorb"))




# sentences = [remove_stops(tokenizer.tokenize(sentence)) for sentence in met_sents+lit_sents]

# print(munge(met_sents[0]["sent"]))
# print(sentences)
# print(metrics(met_sents[0]))

# lit_data = list(zip(*[metrics(s) for s in lit_sents]))
# print(data)

