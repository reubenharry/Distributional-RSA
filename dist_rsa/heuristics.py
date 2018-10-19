import nltk
import scipy
import numpy as np
import pandas as pd
import pickle
import itertools
from dist_rsa.utils.load_data import *
from dist_rsa.utils.helperfunctions import projection
import spacy
nlp = spacy.load('en')
from nltk.corpus import stopwords
import random
import math
from sklearn.linear_model import Perceptron,LogisticRegression

random.seed(4)

stops = set(stopwords.words('english'))
vec_size,vec_kind = 300,'glove.6B.'
vecs = load_vecs(mean=True,pca=False,vec_length=vec_size,vec_type=vec_kind) 

def remove_stops(sent):
	without_stops = [str(w).lower() for w in sent if ((str(w) not in stops) and (len(str(w))>2) )]
	return without_stops

def sent_to_dict(sent,verb_lemma):

	parsed = nlp(sent)
	verb = [word for word in parsed if word.lemma_==verb_lemma][0]
	dobj = [word for word in verb.children if word.dep_=="dobj"]
	nsubj = [word for word in verb.children if (word.dep_=="nsubj" or word.dep_=="nsubjpass")]
	if remove_stops(dobj) == [] and remove_stops(nsubj) == []:
		raise Exception
	if not dobj==[]:
		dobj = list(dobj[0].subtree)
	if not nsubj==[]:
		nsubj = list(nsubj[0].subtree)
	out = {"sent":remove_stops(parsed),"dobj":remove_stops(dobj),"verb":verb_lemma,"nsubj":remove_stops(nsubj)}
	return out

def failable_metrics(sent,verb_lemma,vecs):

	try:
		sent = sent_to_dict(sent,verb_lemma)
	except: 
		return None,None

	projection_words = sorted(list(set([w for w in sent["sent"] if (w in vecs) and (w not in sent["dobj"]+sent["nsubj"]+[sent["verb"]] )])))
	if projection_words == []: return None
	projection_matrix = np.asarray([vecs[w] for w in projection_words]).T

	s_o = np.mean([vecs[v] for v in sent["dobj"]+sent["nsubj"] if v in vecs],axis=0)
	v = vecs[sent["verb"]]

	distance_in_original_space = scipy.spatial.distance.euclidean(v,s_o)

	try:projected_s_o = np.squeeze(projection(np.expand_dims(s_o,1), projection_matrix))
	except: 
		raise Exception
	
	projected_v = np.squeeze(projection(np.expand_dims(v,1), projection_matrix))
	distance_in_subspace = scipy.spatial.distance.euclidean(projected_s_o,projected_v)

	return distance_in_original_space,distance_in_subspace

def metrics_trofi(sent,verb_lemma,vecs):
	try: out = failable_metrics(sent=sent,verb_lemma=verb_lemma,vecs=vecs)
	except: out = (None,None)
	return out

def metrics_tsvetkov(sent_dict,vecs):

	sent = remove_stops(sent_dict["sent"])

	projection_words = sorted(list(set([w for w in sent if (w in vecs) and (w not in [sent_dict["dobj"],sent_dict["nsubj"],sent_dict["verb"]] )])))
	if projection_words == []: return None,None
	projection_matrix = np.asarray([vecs[w] for w in projection_words]).T

	# obj_and_subj = [x for x in [sent_dict["dobj"],sent_dict["nsubj"]] if not np.isNan(x)]

	s_o = np.mean([vecs[v] for v in [sent_dict["dobj"],sent_dict["nsubj"]] if v in vecs],axis=0)
	v = vecs[sent_dict["verb"]]

	distance_in_original_space = scipy.spatial.distance.euclidean(v,s_o)

	try:projected_s_o = np.squeeze(projection(np.expand_dims(s_o,1), projection_matrix))
	except: 
		raise Exception
	
	projected_v = np.squeeze(projection(np.expand_dims(v,1), projection_matrix))
	distance_in_subspace = scipy.spatial.distance.euclidean(projected_s_o,projected_v)




# if __name__ == "__main__":




# write separate pipeline to lit_data,met_data
# wrap what's below as function from lit_data,met_data to output


def produce_tsvetkov_data():

	lit_data_raw,met_data_raw = load_tsvetkov_data()
	
	lit_data = []
	met_data = []
	
	for datum in lit_data:
		lit_data += [metrics_tsvetkov(sent_dict=datum,vecs=vecs)]

	for datum in met_data:
		met_data += [metrics_tsvetkov(sent_dict=datum,vecs=vecs)]

	new_lit_data = [[pair[0],pair[1],0] for pair in lit_data if pair is not None and pair[0] is not None and pair[1] is not None]
	new_met_data = [[pair[0],pair[1],1] for pair in met_data if pair is not None and pair[0] is not None and pair[1] is not None]
	
	return new_lit_data,new_met_data

def produce_trofi_data():

	data = load_trofi_data()
	lit_data = []
	met_data = []
	for verb in sorted(list(data)):

	    lit_data += [metrics_trofi(sent=s,verb_lemma=verb,vecs=vecs) for s in data[verb]["literal"] ]
	    met_data += [metrics_trofi(sent=s,verb_lemma=verb,vecs=vecs) for s in data[verb]["non_literal"]  ]

# new_lit_data = [[pair[0],pair[1][0],pair[1][1]] for pair in lit_data if pair[1] is not None and pair[1][0] is not None and pair[1][1] is not None]
# new_met_data = [[pair[0],pair[1][0],pair[1][1]] for pair in met_data if pair[1] is not None and pair[1][0] is not None and pair[1][1] is not None]

	new_lit_data = [[pair[0],pair[1],0] for pair in lit_data if pair is not None and pair[0] is not None and pair[1] is not None]
	new_met_data = [[pair[0],pair[1],1] for pair in met_data if pair is not None and pair[0] is not None and pair[1] is not None]
	return new_lit_data,new_met_data
# met_data = [pair for pair in met_data if pair is not None and pair[0] is not None and pair[1] is not None]

def data_to_predictions(lit_data,met_data):

	Xy = lit_data+met_data
	random.shuffle(Xy)
	X = np.asarray([[item[0],item[1]] for item in Xy])
	y = np.asarray([[item[2]] for item in Xy])

	len_data = len(X)
	tenth_of_data = len_data // 10
	train_split = 4 * tenth_of_data
	val_split = 5 * tenth_of_data
	test_split = len_data

	train_X = X[:train_split]
	train_y = y[:train_split]
	val_X = X[train_split:val_split]
	val_y = y[train_split:val_split]
	test_X = X[val_split:test_split]
	test_y = y[val_split:test_split]

	unzipped_lit_val_data = list(zip(*[x for (i,x) in enumerate(val_X) if (val_y)[i]==0]))
	unzipped_met_val_data = list(zip(*[x for (i,x) in enumerate(val_X) if (val_y)[i]==1]))

	unzipped_lit_train_data = list(zip(*[x for (i,x) in enumerate(train_X) if (train_y)[i]==0]))
	unzipped_met_train_data = list(zip(*[x for (i,x) in enumerate(train_X) if (train_y)[i]==1]))

	return unzipped_lit_val_data, unzipped_met_val_data, unzipped_lit_train_data, unzipped_met_train_data

if __name__ == "__main__":

	lit_data,met_data = produce_trofi_data()
	unzipped_lit_val_data, unzipped_met_val_data, unzipped_lit_train_data, unzipped_met_train_data = data_to_predictions(lit_data=lit_data,met_data=met_data)
	print(unzipped_lit_train_data)