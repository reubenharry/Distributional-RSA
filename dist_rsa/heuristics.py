import nltk
import scipy
import numpy as np
import pickle
import itertools
from dist_rsa.utils.load_data import *
import spacy
nlp = spacy.load('en')
# from dist_rsa.utils.helperfunctions import *
# from dist_rsa.utils.config import abstract_threshold,concrete_threshold

# find non-stop words
# find matrix verb
# project glove vectors into subspace

sentence = "The man waded through the paperwork."

from nltk.corpus import stopwords
stops = set(stopwords.words('english'))

vec_size,vec_kind = 300,'glove.6B.'
vecs = load_vecs(mean=True,pca=False,vec_length=vec_size,vec_type=vec_kind) 
projection_matrix = np.asarray([vecs["man"],vecs["paperwork"]]).T

print(scipy.spatial.distance.cosine(vecs["waded"],vecs["man"]+vecs["paperwork"]))

projected_man = np.dot(np.expand_dims(vecs["man"],0), projection_matrix)
projected_waded = np.dot(np.expand_dims(vecs["waded"],0), projection_matrix)
projected_paperwork = np.dot(np.expand_dims(vecs["paperwork"],0), projection_matrix)

print(scipy.spatial.distance.cosine(projected_waded,projected_man+projected_paperwork))

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')







# sent = u"Another option will be to try to curb the growth in education and other local assistance , which absorbs 66 % of the state 's budget ./."

def sent_to_dict(sent,verb_lemma):

	parsed = nlp(sent)
	verb = [word for word in parsed if word.lemma_==verb_lemma][0]
	dobj = [word for word in verb.children if word.dep_=="dobj"]
	if dobj == []:
		raise Exception
	dobj = list(dobj[0].subtree)
	out = {"sent":parsed,"dobj":dobj,"verb":verb_lemma}
	print(out)
	return out



def metrics(sent):

	try:sent = sent_to_dict(sent,"absorb")
	except: return None
	# print("begin sent",sent,"end sent")
	# print(sent.keys())

	projection_matrix = np.asarray([vecs[str(w)] for w in sent["sent"] if str(w) in vecs]).T

	# s = np.mean([vecs[v] for v in sent["subj"] if v in vecs],axis=0)
	# print(s.shape,projection_matrix.shape)
	# raise Exception
	o = np.mean([vecs[str(v)] for v in sent["dobj"] if str(v) in vecs],axis=0)
	# print("O",o)
	v = vecs[sent["verb"]]
	# print(o)
	# print(v)
	# s_o = np.mean([s,o],axis=0)

	distance_in_original_space = scipy.spatial.distance.cosine(v,o)


	projected_o = np.dot(np.expand_dims(o,0), projection_matrix)
	projected_v = np.dot(np.expand_dims(v,0), projection_matrix)

	distance_in_subspace = scipy.spatial.distance.cosine(projected_o,projected_v)

	return distance_in_original_space/distance_in_subspace, distance_in_subspace

print(metrics("John absorbed."))




# sentences = [remove_stops(tokenizer.tokenize(sentence)) for sentence in met_sents+lit_sents]

# print(munge(met_sents[0]["sent"]))
# print(sentences)
# print(metrics(met_sents[0]))

# met_data = list(zip(*[metrics(s) for s in met_sents]))
# lit_data = list(zip(*[metrics(s) for s in lit_sents]))
# print(data)

