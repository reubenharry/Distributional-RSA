from __future__ import division
# name = input("save to: ")

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


vec_size,vec_kind = 200,'glove.twitter.27B.'
real_vecs = load_vecs(mean=True,pca=True,vec_length=vec_size,vec_type=vec_kind)  
vecs = real_vecs
nouns,adjs = get_words(with_freqs=False)
freqs = pickle.load(open('dist_rsa/data/google_freqs/freqs','rb'))

qud_words = [a for a in list(adjs) if a in vecs and a in real_vecs]
quds = sorted(qud_words,key=lambda x:freqs[x],reverse=True)

possible_utterance_nouns = [n for n in nouns if nouns[n] > concrete_threshold and n in vecs and n in real_vecs]
possible_utterances = sorted(possible_utterance_nouns,key=lambda x:freqs[x],reverse=True)
possible_utterances=possible_utterances[:100]

quds = quds[:700]
print(quds[:700])

def l1_model(metaphor):
    subj,pred,sig1,sig2,is_baseline = metaphor

    print('abstract_threshold',abstract_threshold)
    print('concrete_threshold',concrete_threshold)


    # qud_words = [a for a in list(adjs) if adjs[a] < abstract_threshold and a in vecs and a in real_vecs]
    # quds_near_subj = sorted(qud_words,\
    #     key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[subj]],axis=0)),reverse=False)

    # quds_near_pred = sorted(qud_words,\
    #         key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[pred],vecs[pred]],axis=0)),reverse=False)

    # possible_utterance_nouns = sorted([n for n in nouns if nouns[n] > concrete_threshold and n in vecs],\
    #     key=lambda x: scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[subj]],axis=0)),reverse=False)

    # quds = quds[:200]
    # quds = sorted(quds,key=lambda x: adjs[x][1],reverse=True)
    # quds = quds_near_pred[:32]+quds_near_subj[:32]
    # quds = list(set(quds))

    # possible_utterance_adjs = quds
    # print("QUDS",quds[:50]) 
    # possible_utterances = possible_utterance_adjs+possible_utterance_nouns[:100]

    # possible_utterances = ['ox','bag','nightmare']
    # possible_utterances = possible_utterance_adjs[:50]
    # +quds[:25]
    # possible_utterance_adjs[:50]+possible_utterance_nouns[:50]
    # possible_utterance_nouns[:4]


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
        sample_number = 1000,
        number_of_qud_dimensions=1,
        # burn_in=900,
        seed=False,trivial_qud_prior=False,
        step_size=1e-5,
        poss_utt_frequencies=defaultdict(lambda:1),
        qud_frequencies=defaultdict(lambda:1),
        qud_prior_weight=0.5,
        rationality=1.0,
        norm_vectors=False,
        variational=True,
        variational_steps=300,
        baseline=is_baseline
        # world_movement=True
        )

    run = Dist_RSA_Inference(params)
    run.compute_l1(load=0,save=False)
    results = run.qud_results()

    print(results[:5])

    if not is_baseline:
        worldm = run.world_movement("cosine",comparanda=[x for x in qud_words if x in real_vecs])
        print("\nworld\n",worldm[:5])
    else: worldm = None
        # out.write("\nWORLD MOVEMENT:\n")
        # out.write(str(worldm))
    # print("WORLD MOVEMENT WITH PROJECTION\n:",run.world_movement("cosine",comparanda=[x for x in quds if x in real_vecs],do_projection=True)[:50])
    # print("BASELINE:\n",sorted(qud_words,\
    #     key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[pred]],axis=0)),reverse=False)[:5])




if __name__ == "__main__":

    # l1_dict = {}
    # qud_only_dict = {}

    # out = open("dist_rsa/data/l1_results_"+name,"w")
    # out.write("RESULTS 25D\n")
    for subj,pred in [("man","lion"),("woman","rose"),("place","junkyard")]:
        l1_model((subj,pred,0.1,0.1,True))

        l1_model((subj,pred,10.0,0.01,False))
        l1_model((subj,pred,10.0,0.01,False))


                        # print(l1_dict)


    # pickle.dump(l1_dict,open("l1_dict"+name,"wb"))
    # pickle.dump(qud_only_dict,open("qud_only_dict"+name,"wb"))


