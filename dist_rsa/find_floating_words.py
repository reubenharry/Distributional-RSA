from __future__ import division

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


vecs = load_vecs(mean=True,pca=True,vec_length=300,vec_type='glove.6B.')
nouns,adjs = get_words(with_freqs=True)

qud_words = [a for a in list(adjs) if  a in vecs]
quds = sorted(qud_words,key = lambda x: adjs[x][1],reverse=True)
quds = quds[:100]

def l1_model(metaphor):
    subj,pred,sig1,sig2,is_baseline = metaphor
    
    vec_size,vec_kind = 50,'glove.6B.'

    print('abstract_threshold',abstract_threshold)
    print('concrete_threshold',concrete_threshold)

# adjs[a][0] < abstract_threshold and

    # prob_dict = get_freqs(preprocess=False)
    # prob_dict = predict(" ".join([subj, "is","a"]))
    
        # key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[pred]],axis=0)),reverse=False)
        # key=lambda x:prob_dict[x],reverse=True)


    possible_utterance_nouns = sorted([n for n in nouns if n in vecs],\
        # key=lambda x:prob_dict[x],reverse=True)
        key = lambda x: nouns[x][1],reverse=True)
        # key=lambda x: scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[subj]],axis=0)),reverse=False)
    # possible_utterance_nouns = 
    # break
    possible_utterance_adjs = quds
    print("QUDS",quds[:50]) 
    possible_utterances = possible_utterance_nouns[:100]
    # +possible_utterance_adjs[:10]

    # possible_utterances = ['ox','bag','nightmare']
    # possible_utterances = possible_utterance_adjs[:50]
    # +quds[:25]
    # possible_utterance_adjs[:50]+possible_utterance_nouns[:50]
    # possible_utterance_nouns[:4]

    real_vecs = load_vecs(mean=True,pca=True,vec_length=vec_size,vec_type=vec_kind)    

    for x in possible_utterances:
        if x not in real_vecs:
            print(x,"not in vecs")
            possible_utterances.remove(x)
            # raise Exception("utterance not in vecs")

    print("UTTERANCES:\n",possible_utterances[:20])

    params = Inference_Params(
        vecs=real_vecs,
        subject=[subj],predicate=pred,
        quds=quds,
        possible_utterances=list(set(possible_utterances).union(set([pred]))),
        sig1=sig1,sig2=sig2,
        qud_weight=0.0,freq_weight=0.0,
        categorical="categorical",
        sample_number = 100,
        number_of_qud_dimensions=1,
        # burn_in=900,
        seed=False,trivial_qud_prior=False,
        step_size=1e-2,
        poss_utt_frequencies=defaultdict(lambda:1),
        qud_frequencies=defaultdict(lambda:1),
        qud_prior_weight=0.5,
        rationality=0.99,
        norm_vectors=False,
        variational=True,
        variational_steps=100,
        baseline=is_baseline
        # world_movement=True

        )

    run = Dist_RSA_Inference(params)

    run.compute_l1(load=0,save=False)

    results = run.qud_results()

    # print(results[:50])

    return results

if __name__ == "__main__":

    # for each word, count how many top results it appears in
    word_count = defaultdict(list)

    for subj,pred in metaphors:
        word_occured = defaultdict(lambda:False)

        for sig1,sig2 in [(0.1,0.1),(1.0,0.1),(100.0,0.1),(0.1,0.001),(0.0001,0.1),(0.00001,0.00001)]:
            for is_baseline in [False]:
                # if is_baseline:
                #     out.write('\n\nBASELINE')
                #     results = l1_model((subj,pred,sig1,sig2,is_baseline))
                #     # results = [(0.5,0.5)]
                #     out.write(str([(x,np.exp(y)) for (x,y) in results[:5]]))
                # else:
                for i in range(3):
                    results = l1_model((subj,pred,sig1,sig2,is_baseline))
                    word = results[0][0][0]
                    if word_occured[word]==False:
                        word_occured[word]==True
                        word_count[word]+=[(subj,pred)]
                    # results = [(0.5,0.5)]
            out = sorted([(q,set(word_count[q])) for q in quds],key=lambda x : len(x[1]),reverse=True)
            print(out)
            pickle.dump(out,open("dist_rsa/data/word_counts",'wb'))

    print(sorted([(q,set(word_count[q])) for q in quds],key=lambda x : len(x[1]),reverse=True))

