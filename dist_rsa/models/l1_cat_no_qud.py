from __future__ import division
from collections import defaultdict
import scipy
import numpy as np
import pickle
import itertools
from dist_rsa.dbm import *
from dist_rsa.utils.load_data import *
from dist_rsa.utils.helperfunctions import *
from dist_rsa.utils.load_data import get_words,get_freqs
from dist_rsa.lm_1b_eval import predict


vecs = load_vecs(mean=True,pca=True,vec_length=300,vec_type='glove.6B.')
nouns,adjs = get_words()

def l1_cat_2d_exp(metaphor):
    vec_size,vec_kind = 25,'glove.twitter.27B.'
    subj,pred = metaphor
    abstract_threshold = 2.5
    print('abstract_threshold',abstract_threshold)
    concrete_threshold = 3.0
    print('concrete_threshold',concrete_threshold)

    qud_words = [a for a in list(adjs) if adjs[a] < abstract_threshold and a in vecs]

    sig2_distance = scipy.spatial.distance.cosine(vecs[subj],vecs[pred])

    # prob_dict = get_freqs(preprocess=False)
    # predict(" ".join([subj, "is","a"]))
    
    quds = sorted(qud_words,\
        key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[pred]],axis=0)),reverse=False)
        # key=lambda x:prob_dict[x],reverse=True)


    possible_utterance_nouns = sorted([n for n in nouns if nouns[n] > concrete_threshold and n in vecs],\
        key=lambda x: scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[subj]],axis=0)),reverse=False)
        # key=lambda x:prob_dict[x],reverse=True)
    # possible_utterance_nouns = 
    # break
    possible_utterance_adjs = quds
    quds = quds[:50]
    possible_utterances = animals[:10]
    # possible_utterance_adjs[:20]+possible_utterance_nouns[:20]
    # possible_utterance_nouns[:4]


    print("UTTERANCES:\n",possible_utterances[:20])

    real_vecs = load_vecs(mean=True,pca=True,vec_length=vec_size,vec_type=vec_kind)    
    params = Inference_Params(
        vecs=real_vecs,
        subject=[subj],predicate=pred,
        quds=quds,
        possible_utterances=list(set(possible_utterances).union(set([pred]))),
        sig1=0.001,sig2=0.01,
        qud_weight=0.0,freq_weight=0.0,
        categorical="categorical",
        sample_number = 50000,
        number_of_qud_dimensions=2,
        # burn_in=900,
        seed=False,trivial_qud_prior=False,
        step_size=0.0005,
        poss_utt_frequencies=defaultdict(lambda:1),
        qud_frequencies=defaultdict(lambda:1),
        qud_prior_weight=0.5,
        rationality=0.9,
        norm_vectors=False,
        variational=True,
        variational_steps=50000,
        only_trivial=True
        )

    run = Dist_RSA_Inference(params)

    run.compute_l1(load=0,save=False)

    # print(run.world_samples.shape)

    # results = run.qud_results()

    # print(results[:20])
    # run.compute_s1(params,s1_world=)

    print("WORLD MOVEMENT\n:",run.world_movement("cosine",comparanda=[x for x in quds if x in real_vecs])[:50])
    print("WORLD MOVEMENT WITH PROJECTION\n:",run.world_movement("cosine",comparanda=[x for x in quds if x in real_vecs],do_projection=True)[:50])
    # print("BASELINE:\n",sorted(qud_words,\
    #     key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[pred]],axis=0)),reverse=False)[:20])

    # print("RESULTS\n",[(x,np.exp(y)) for (x,y) in results[:20]])

    # return results

if __name__ == "__main__":

    l1_cat_2d_exp(("man","ox"))
    l1_cat_2d_exp(("man","ox"))

    # l1_cat_2d_exp(("bed","heaven"))
    # l1_cat_2d_exp(("bed","heaven"))
    # l1_cat_2d_exp(("woman","rose"))
    # l1_cat_2d_exp(("woman","rose"))
    # l1_cat_2d_exp(("flower","rose"))
    # l1_cat_2d_exp(("flower","rose"))
    # l1_cat_2d_exp(("woman","car"))
    # l1_cat_2d_exp(("woman","car"))
    # l1_cat_2d_exp(("rose","woman"))
    # l1_cat_2d_exp(("rose","woman"))
