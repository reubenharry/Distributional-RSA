import nltk
import scipy
import numpy as np
import pandas as pd
import pickle
import itertools
from dist_rsa.utils.load_data import *
from dist_rsa.utils.helperfunctions import projection
import spacy
from nltk.corpus import stopwords
import random
import math
from sklearn.linear_model import Perceptron,LogisticRegression
from nltk.stem.wordnet import WordNetLemmatizer

<<<<<<< HEAD
nlp = spacy.load('en')
random.seed(4)
=======
ps = nltk.stem.PorterStemmer()

# lmtzr = WordNetLemmatizer()

# random.seed(4)
>>>>>>> 83928f8975223c75ca7ebf5ecd39f6bd4350bb13

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

def failable_metrics(sent,verb_lemma,vecs,our_method):

	try:
		sent = sent_to_dict(sent,verb_lemma)
	except: 
		return None,None

	projection_words = sorted(list(set([w for w in sent["sent"] if (w in vecs) and (w not in sent["dobj"]+sent["nsubj"]+[sent["verb"]] )])))
	if projection_words == []: return None
	projection_matrix = np.asarray([vecs[w] for w in projection_words]).T

	print(sent)
	print(projection_words)

	s_o = np.mean([vecs[v] for v in sent["dobj"]+sent["nsubj"] if v in vecs],axis=0)
	v = vecs[sent["verb"]]

	distance_in_original_space = scipy.spatial.distance.euclidean(v,s_o)

	try:projected_s_o = np.squeeze(projection(np.expand_dims(s_o,1), projection_matrix))
	except: 
		raise Exception
	
	projected_v = np.squeeze(projection(np.expand_dims(v,1), projection_matrix))
	distance_in_subspace = scipy.spatial.distance.euclidean(projected_s_o,projected_v)

	return distance_in_original_space,distance_in_subspace

def metrics_trofi(sent,verb_lemma,vecs,our_method):
	print("NEXT TROFI ITEM")
	try: out = failable_metrics(sent=sent,verb_lemma=verb_lemma,vecs=vecs,our_method=our_method)
	except: out = (None,None)
	return out

def metrics_tsvetkov(sent_dict,vecs,our_method=True):

	sent = remove_stops(nltk.word_tokenize(sent_dict["sent"]))

	# print("sent",sent_dict["sent"])
	# print("sent",sent)
	# print("tings")
	# print(sent_dict["dobj"])
	# print(type(sent_dict["dobj"])==str)
	# print(sent_dict["nsubj"])
	# print(type(sent_dict["nsubj"])==str)
	# print(sent_dict["verb"])
	# print(type(sent_dict["verb"])==str)
	# np.isnan(sent_dict["verb"])

	# raise Exception
	# condition = ((type(sent_dict["dobj"])==str) or (type(sent_dict["nsubj"])==str)) and (type(sent_dict["verb"])==str)
	# print(condition)

	lemmatized = [ps.stem(w) for w in sent]
	# lemmatized = [sent_dict["dobj"],sent_dict["nsubj"],sent_dict["verb"]]
	projection_words = sorted(list(set([w for w in lemmatized if (w in vecs) and (w not in [sent_dict["dobj"],sent_dict["nsubj"],sent_dict["verb"]] )])))
	if sent_dict["verb"] not in lemmatized:
		return None,None
	# print("SENT_DICT",sent_dict)
	# print("PROJECTION WORDS",projection_words)
	if projection_words == []: 
		print("NO PROJECTION WORDS")
		return None,None
	projection_matrix = np.asarray([vecs[w] for w in projection_words]).T

	# obj_and_subj = [x for x in [sent_dict["dobj"],sent_dict["nsubj"]] if not np.isNan(x)]

	if not (type(sent_dict["nsubj"])==str):
		s_o = vecs[sent_dict["dobj"]]

	elif not (type(sent_dict["dobj"])==str):
		s_o = vecs[sent_dict["nsubj"]]

	else:
		s_o = np.mean([vecs[v] for v in [sent_dict["dobj"],sent_dict["nsubj"]] if v in vecs],axis=0)
	
	v = vecs[sent_dict["verb"]]

	distance_in_original_space = scipy.spatial.distance.euclidean(v,s_o)

	try:projected_s_o = np.squeeze(projection(np.expand_dims(s_o,1), projection_matrix))
	except: 
		raise Exception
	
	projected_v = np.squeeze(projection(np.expand_dims(v,1), projection_matrix))
	distance_in_subspace = scipy.spatial.distance.euclidean(projected_s_o,projected_v)




	s_o_distance = scipy.spatial.distance.cosine(projected_s_o,s_o)
	v_distance = scipy.spatial.distance.cosine(projected_v,v)

	if our_method:
		return distance_in_original_space,distance_in_subspace
	else:
		return s_o_distance,v_distance
	# feature_1 = 
	# feature_2 =
	# feature_3 =
	# feature_4 = 

# if __name__ == "__main__":




# write separate pipeline to lit_data,met_data
# wrap what's below as function from lit_data,met_data to output


def produce_tsvetkov_data(our_method):

	lit_data_raw,met_data_raw = load_tsvetkov_data()
	
	lit_data = []
	met_data = []

	print("lengths orig",len(lit_data_raw),len(met_data_raw))
	
	for datum in lit_data_raw:
		# condition_1 = ((type(datum["dobj"])==str) or (type(datum["nsubj"])==str)) and (type(datum["verb"])==str)
		condition_2 = (datum["dobj"] in vecs or not type(datum["dobj"])==str) and (datum["nsubj"] in vecs or not type(datum["nsubj"])==str) and (datum["verb"] in vecs or not type(datum["verb"])==str) 
		if condition_2:

			# print("NEXT ITEM")
			lit_data += [metrics_tsvetkov(sent_dict=datum,vecs=vecs,our_method=our_method)]

	for datum in met_data_raw:
		# condition_1 = ((type(datum["dobj"])==str) or (type(datum["nsubj"])==str)) and (type(datum["verb"])==str)
		condition_2 = (datum["dobj"] in vecs or not type(datum["dobj"])==str) and (datum["nsubj"] in vecs or not type(datum["nsubj"])==str) and (datum["verb"] in vecs or not type(datum["verb"])==str) 
		if condition_2:
			# print(datum,"p")
			met_data += [metrics_tsvetkov(sent_dict=datum,vecs=vecs,our_method=our_method)]

	new_lit_data = [[pair[0],pair[1],0] for pair in lit_data if pair is not None and pair[0] is not None and pair[1] is not None]
	new_met_data = [[pair[0],pair[1],1] for pair in met_data if pair is not None and pair[0] is not None and pair[1] is not None]
	
	print("lengths",len(new_lit_data),len(new_met_data))

	return new_lit_data,new_met_data

def produce_trofi_data(our_method=True):

	data = load_trofi_data()
	lit_data = []
	met_data = []
	for verb in sorted(list(data)):



	    lit_data += [metrics_trofi(sent=s,verb_lemma=verb,vecs=vecs,our_method=our_method) for s in data[verb]["literal"] ]
	    met_data += [metrics_trofi(sent=s,verb_lemma=verb,vecs=vecs,our_method=our_method) for s in data[verb]["non_literal"]  ]

# new_lit_data = [[pair[0],pair[1][0],pair[1][1]] for pair in lit_data if pair[1] is not None and pair[1][0] is not None and pair[1][1] is not None]
# new_met_data = [[pair[0],pair[1][0],pair[1][1]] for pair in met_data if pair[1] is not None and pair[1][0] is not None and pair[1][1] is not None]

	new_lit_data = [[pair[0],pair[1],0] for pair in lit_data if pair is not None and pair[0] is not None and pair[1] is not None]
	new_met_data = [[pair[0],pair[1],1] for pair in met_data if pair is not None and pair[0] is not None and pair[1] is not None]
	return new_lit_data,new_met_data
# met_data = [pair for pair in met_data if pair is not None and pair[0] is not None and pair[1] is not None]

def data_to_predictions(lit_data,met_data):

	Xy = lit_data+met_data
	random.seed(4)
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

	return unzipped_lit_val_data, unzipped_met_val_data, unzipped_lit_train_data, unzipped_met_train_data,train_X,train_y,val_X,val_y,test_X,test_y

if __name__ == "__main__":

	lit_data,met_data = produce_trofi_data(our_method=True)

	# lit_data,met_data = produce_tsvetkov_data(our_method=False)
	print(len(lit_data),len(met_data))

	unzipped_lit_val_data, unzipped_met_val_data, unzipped_lit_train_data, unzipped_met_train_data, train_X,train_y,val_X,val_y,test_X,test_y = data_to_predictions(lit_data=lit_data,met_data=met_data)
	print(unzipped_lit_train_data[0][0])
	# print(unzipped_lit_train_data) 