from __future__ import division
from collections import defaultdict
import scipy
import numpy as np
import pickle
import itertools
from dist_rsa.dbm import *
from dist_rsa.utils.load_data import *
from dist_rsa.utils.helperfunctions import *
from dist_rsa.utils.load_data import get_words
from dist_rsa.lm_1b_eval import predict
from dist_rsa.utils.config import abstract_threshold,concrete_threshold


vecs = load_vecs(mean=True,pca=True,vec_length=300,vec_type='glove.6B.')
nouns,adjs = get_words()

def l1_cat_2d_exp(metaphor):
    vec_size,vec_kind = 25,'glove.twitter.27B.'
    subj,pred = metaphor

    print('abstract_threshold',abstract_threshold)
    print('concrete_threshold',concrete_threshold)

    qud_words = [a for a in list(adjs) if adjs[a] < abstract_threshold and a in vecs]


    # prob_dict = get_freqs(preprocess=False)
    # prob_dict = predict(" ".join([subj, "is","a"]))
    
    quds = sorted(qud_words,\
        key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[pred]],axis=0)),reverse=False)
        # key=lambda x:prob_dict[x],reverse=True)


    possible_utterance_nouns = sorted([n for n in nouns if nouns[n] > concrete_threshold and n in vecs],\
        # key=lambda x:prob_dict[x],reverse=True)
        key=lambda x: scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[subj]],axis=0)),reverse=False)
    # possible_utterance_nouns = 
    # break
    possible_utterance_adjs = quds
    quds = quds[:50]
    print("QUDS",quds[:50]) 
    possible_utterances = possible_utterance_nouns[:200:5]+possible_utterance_adjs[:40]

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
        sig1=0.001,sig2=0.01,
        qud_weight=0.0,freq_weight=0.0,
        categorical="categorical",
        sample_number = 500,
        number_of_qud_dimensions=2,
        # burn_in=900,
        seed=False,trivial_qud_prior=False,
        step_size=1e-3,
        poss_utt_frequencies=defaultdict(lambda:1),
        qud_frequencies=defaultdict(lambda:1),
        qud_prior_weight=0.5,
        rationality=0.9,
        norm_vectors=False,
        variational=True,
        variational_steps=20,
        # world_movement=True

        )

    run = Dist_RSA_Inference(params)

    run.compute_l1(load=0,save=False)

    # print(run.world_samples.shape)

    results = run.qud_results()

    # print(results[:20])
    # run.compute_s1(params,s1_world=)

    print("WORLD MOVEMENT\n:",run.world_movement("cosine",comparanda=[x for x in quds if x in real_vecs])[:50])
    print("WORLD MOVEMENT WITH PROJECTION\n:",run.world_movement("cosine",comparanda=[x for x in quds if x in real_vecs],do_projection=True)[:50])
    print("BASELINE:\n",sorted(qud_words,\
        key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[pred]],axis=0)),reverse=False)[:20])

    print("RESULTS\n",[(x,np.exp(y)) for (x,y) in results[:20]])

    return results

if __name__ == "__main__":

    # l1_cat_2d_exp(("love","poison"))
    # l1_cat_2d_exp(("woman","rose"))
    l1_cat_2d_exp(("man","lion"))
    l1_cat_2d_exp(("man","lion"))
    l1_cat_2d_exp(("man","ox"))
    l1_cat_2d_exp(("man","ox"))
    l1_cat_2d_exp(("voice","river"))
    l1_cat_2d_exp(("voice","river"))
    # l1_cat_2d_exp(("man","ox"))
    # l1_cat_2d_exp(("man","ox"))
    # # l1_cat_2d_exp(("man","lion"))

    # # l1_cat_2d_exp(("bed","heaven"))
    # # l1_cat_2d_exp(("bed","heaven"))

    # # print(scipy.spatial.distance.cosine(vecs['man'],vecs['lion']))
    # # print(scipy.spatial.distance.cosine(vecs['bed'],vecs['heaven']))
    # l1_cat_2d_exp(("heaven","bed"))
    # l1_cat_2d_exp(("heaven","bed"))
    # # l1_cat_2d_exp(("woman","rose"))
    # l1_cat_2d_exp(("flower","rose"))
    # l1_cat_2d_exp(("flower","rose"))
    # l1_cat_2d_exp(("woman","car"))
    # l1_cat_2d_exp(("woman","car"))
    # l1_cat_2d_exp(("rose","woman"))
    # l1_cat_2d_exp(("rose","woman"))
