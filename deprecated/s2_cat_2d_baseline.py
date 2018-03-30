from dist_rsa.utils.load_data import get_words,load_vecs
import numpy as np
import scipy
from dist_rsa.utils.config import concrete_threshold

vecs = load_vecs(mean=True,pca=True,vec_length=300,vec_type='glove.6B.')
nouns,adjs = get_words()

subj = 'man'

possible_utterance_nouns = sorted([n for n in nouns if nouns[n] > concrete_threshold and n in vecs],\
    # key=lambda x:prob_dict[x],reverse=True)
    key=lambda x: scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[subj]],axis=0)),reverse=False)

possible_utterances = possible_utterance_nouns[:100]

print(possible_utterances)

q1 = 'stupid'
q2 = 'obsessed'


out = sorted(possible_utterances,key=lambda x : scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[q1],vecs[q2]],axis=0)),reverse=False)

print('\n',out)
# def s2_baseline(self):

    # mean of the three words


