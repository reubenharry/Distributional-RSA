from collections import defaultdict
import scipy
import numpy as np
import pickle
import itertools
from dist_rsa.dbm import *
# from dist_rsa.utils.load_data import *
from dist_rsa.utils.helperfunctions import *
from dist_rsa.lm_1b_eval import predict
from dist_rsa.utils.config import abstract_threshold,concrete_threshold
# from dist_rsa.utils.distance_under_projection import distance_under_projection
import edward as ed
from dist_rsa.utils.simple_vecs import real_vecs as simple_vecs




def l1_model(subj,pred,sig1,sig2,l1_sig1,resolution,quds,only_trivial):
    vec_size,vec_kind = 25,'glove.twitter.27B.'

    # print('abstract_threshold',abstract_threshold)
    # print('concrete_threshold',concrete_threshold)

    # quds = ['old','strong','stupid',"fast","green"]
    # possible_utterances = ["wall","tree","man","idiot"]
    # possible_utterances = ['ox','bag','nightmare']
    # possible_utterances = possible_utterance_adjs[:50]
    # +quds[:25]
    # possible_utterance_adjs[:50]+possible_utterance_nouns[:50]
    # possible_utterance_nouns[:4]

    vecs = simple_vecs
    real_vecs= simple_vecs

    # real_vecs = load_vecs(mean=True,pca=False,vec_length=vec_size,vec_type=vec_kind)
    # real_vecs['subj1']=real_vecs["pebble"]
    # real_vecs['subj2']=real_vecs["many"]
    # real_vecs['pred1']=real_vecs["myth"]

    # vecs['subj1']=vecs["many"]
    # vecs['subj2']=vecs["pebble"]
    # vecs['pred1']=vecs["myth"]


    # quds = list(adjs)[:20]
    # possible_utterances = list(nouns)[:200]  
        

    possible_utterances = ["pred1","pred2"]


    # print("unyielding in real vecs","unyielding" in real_vecs)

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
        possible_utterances=sorted(list(set(possible_utterances).union(set([pred])))),
        sig1=sig1,sig2=sig2, l1_sig1=l1_sig1,
        qud_weight=0.0,freq_weight=0.0,
        categorical="categorical",
        sample_number = 2000,
        number_of_qud_dimensions=1,
        burn_in=1000,
        seed=False,trivial_qud_prior=0.5,
        step_size=1e-1,
        poss_utt_frequencies=defaultdict(lambda:1),
        qud_frequencies=defaultdict(lambda:1),
        qud_prior_weight=0.5,
        rationality=1.0,
        norm_vectors=False,
        variational=False,
        variational_steps=100,
        baseline=False,
        discrete_l1=True,
        resolution=resolution,
        only_trivial=True

        # world_movement=True

        )

    run = Dist_RSA_Inference(params)

    run.compute_l1(load=0,save=False)

    world_samples = run.world_samples

    print(world_samples,world_samples.shape)

    pickle.dump(world_samples,open("dist_rsa/data/heatmap_samples.pkl",'wb'))

    return world_samples

if __name__ == "__main__":

    l1_model(subj="subj1",pred="pred2",sig1=0.1,sig2=0.1,l1_sig1=10.0,resolution=(100,0.1),quds=['qud1','qud2'],only_trivial=True)

