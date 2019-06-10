from __future__ import division
name = input("save to: ")

from collections import defaultdict
import scipy
import numpy as np
import pickle
import itertools
from dist_rsa.dbm import *
from dist_rsa.utils.load_data import *
from dist_rsa.utils.helperfunctions import *
from dist_rsa.lm_1b_eval import predict
from dist_rsa.utils.config import abstract_threshold,concrete_threshold
from dist_rsa.utils.distance_under_projection import distance_under_projection


vecs = load_vecs(mean=False,pca=False,vec_length=300,vec_type='word2vec')
vec_size,vec_kind = 200,'glove.twitter.27B.'
# freqs = pickle.load(open('dist_rsa/data/google_freqs/freqs','rb'))
nouns,adjs = get_words(with_freqs=False)
real_vecs = load_vecs(mean=True,pca=True,vec_length=vec_size,vec_type=vec_kind)  

# qud_words = [a for a in list(adjs) if a in vecs and a in real_vecs]
# quds = sorted(qud_words,key=lambda x:freqs[x],reverse=True)

# possible_utterance_nouns = [n for n in nouns if nouns[n] > concrete_threshold and n in vecs and n in real_vecs]
# possible_utterances = sorted(possible_utterance_nouns,key=lambda x:freqs[x],reverse=True)
# possible_utterances=possible_utterances[:100]

# quds = quds[:70]
# print(quds[:100])

def baseline(metaphor):
    subj,pred = metaphor

    print('abstract_threshold',abstract_threshold)
    print('concrete_threshold',concrete_threshold)

    qud_words = [a for a in list(adjs) if adjs[a] < abstract_threshold and a in vecs]

    quds = sorted(qud_words,\
        key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[pred]],axis=0)),reverse=False)
        # key=lambda x:prob_dict[x],reverse=True)


    return quds
if __name__ == "__main__":

    vector_baseline_dict = {}

    # out = open("dist_rsa/data/l1_results_"+name,"w")
    # out.write("RESULTS 25D\n")
    for met in test_metaphors:
    # [("man","lion")]:
    # ,("lion","man"),("woman","lion"),("woman","rose"),("love","poison")]:
    # ,("woman","lion"),("woman","rose"),("place","junkyard")]:
        vector_baseline_dict[met] = baseline(met)
    # test_metaphors:
    # 
        # out.write("\n"+subj+" is a "+pred)

       


    pickle.dump(vector_baseline_dict,open("vector_baseline_dict_"+name,"wb"))


