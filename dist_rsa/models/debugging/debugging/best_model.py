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
import random

random.seed(9001)

vecs = load_vecs(mean=False,pca=False,vec_length=300,vec_type='word2vec')
vec_size,vec_kind = 25,'glove.twitter.27B.'
freqs = pickle.load(open('dist_rsa/data/google_freqs/freqs','rb'))
nouns,adjs = get_words(with_freqs=False)
real_vecs = load_vecs(mean=True,pca=True,vec_length=vec_size,vec_type=vec_kind)  

qud_words = [a for a in list(adjs) if adjs[a] < abstract_threshold and a in vecs and a in real_vecs]



qud_words = sorted(qud_words,\
#     key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[pred]],axis=0)),reverse=False)
    key=lambda x:freqs[x],reverse=True)

noun_words = [n for n in nouns if nouns[n] > concrete_threshold and n in vecs and n in real_vecs]
noun_words = sorted(noun_words,\
    key=lambda x:freqs[x],reverse=True)

# print("QUDS AND NOUNS 1",qud_words[0],noun_words[0])

random.shuffle(qud_words)
random.shuffle(noun_words)
# print("QUDS AND NOUNS after shuffle",qud_words[0],noun_words[0])


#     key=lambda x: scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[subj]],axis=0)),reverse=False)
# possible_utterance_nouns = 
# break

# qud_words = [a for a in list(adjs) if a in vecs and a in real_vecs]
# quds = sorted(qud_words,key=lambda x:freqs[x],reverse=True)

# possible_utterance_nouns = [n for n in nouns if nouns[n] > concrete_threshold and n in vecs and n in real_vecs]
# possible_utterances = sorted(possible_utterance_nouns,key=lambda x:freqs[x],reverse=True)
# possible_utterances=possible_utterances[:100]

# quds = quds[:70]
# print(quds[:100])

def l1_model(metaphor):
    subj,pred,sig1,sig2,l1_sig1,start,stop,is_baseline,qud_num = metaphor

    quds = qud_words[:qud_num] + ['ugly','strong']
    possible_utterance_adjs = quds
    possible_utterances = noun_words[start:stop]



    print('abstract_threshold',abstract_threshold)
    print('concrete_threshold',concrete_threshold)


    # +possible_utterance_adjs
    print("ARE SUBJ AND PRED IN VECS?",subj in real_vecs,pred in real_vecs)

    for x in possible_utterances:
        if x not in real_vecs:
            # print(x,"not in vecs")
            possible_utterances.remove(x)
            # raise Exception("utterance not in vecs")

    print("QUDS",quds[:50]) 
    print("UTTERANCES:\n",possible_utterances[:50])

    random.shuffle(quds)
    random.shuffle(possible_utterances)

    params = Inference_Params(
        vecs=real_vecs,
        subject=[subj],predicate=pred,
        quds=quds,
        possible_utterances=list(set(possible_utterances).union(set([pred]))),
        sig1=sig1,sig2=sig2,l1_sig1=l1_sig1,
        qud_weight=0.0,freq_weight=0.0,
        categorical="categorical",
        sample_number = 20000,
        number_of_qud_dimensions=1,
        # burn_in=90,
        seed=False,trivial_qud_prior=False,
        step_size=1e-7,
        poss_utt_frequencies=defaultdict(lambda:1),
        qud_frequencies=defaultdict(lambda:1),
        qud_prior_weight=1.0,
        rationality=1.0,
        norm_vectors=False,
        variational=True,
        variational_steps=1000,
        baseline=is_baseline
        # world_movement=True
        )

    run = Dist_RSA_Inference(params)
    run.compute_l1(load=0,save=False)
    results = run.qud_results()

    # print(results[:5])

    if not is_baseline:
        worldm = run.world_movement("cosine",comparanda=[x for x in qud_words if x in real_vecs])
        # print("\nworld\n",worldm[:5])
    else: worldm = None
        # out.write("\nWORLD MOVEMENT:\n")
        # out.write(str(worldm))
    # print("WORLD MOVEMENT WITH PROJECTION\n:",run.world_movement("cosine",comparanda=[x for x in quds if x in real_vecs],do_projection=True)[:50])
    # print("BASELINE:\n",sorted(qud_words,\
    #     key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[pred]],axis=0)),reverse=False)[:5])

    print("RESULTS\n",[(x,np.exp(y)) for (x,y) in results[:5]])
    demarg = demarginalize_product_space(results)
    # print("\ndemarginalized:\n,",demarg[:5])
    # out.write("\ndemarginalized:\n")
    # out.write((str(demarg)))

    # params.number_of_qud_dimensions=1
    # run = Dist_RSA_Inference(params)
    # run.compute_l1(load=0,save=False)
    # results2 = run.qud_results()
    # # print("\n1d results\n",results2[:10])
    # one_d = results2
    # one_d=None



    return [(x,np.exp(y)) for (x,y) in results],worldm

