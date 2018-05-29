from __future__ import division
from collections import defaultdict
import scipy
import numpy as np
import pickle
import itertools
from dist_rsa.dbm import *
from dist_rsa.utils.load_data import *
from dist_rsa.utils.helperfunctions import *
from dist_rsa.utils.config import abstract_threshold,concrete_threshold


vecs = load_vecs(mean=False,pca=False,vec_length=300,vec_type='word2vec')
nouns,adjs = get_words(with_freqs=False)

  
# quds = sorted(qud_words,\
#     key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[pred]],axis=0)),reverse=False)
    # key=lambda x:freqs[x],reverse=True)

noun_words = [n for n in nouns if nouns[n] > concrete_threshold and n in vecs]
possible_utterance_nouns = sorted(noun_words,\
    # key=lambda x:freqs[x],reverse=True)
    key=lambda x: scipy.spatial.distance.cosine(vecs[x],np.mean([vecs['big'],vecs['dog']],axis=0)),reverse=False)
# possible_utterance_nouns = 
# break
print(possible_utterance_nouns[:5])

possible_utterance_nouns = sorted(noun_words,\
    # key=lambda x:freqs[x],reverse=True)
    key=lambda x: scipy.spatial.distance.cosine(vecs[x],np.mean([vecs['sky'],vecs['earth']],axis=0)),reverse=False)

print(possible_utterance_nouns[:5])