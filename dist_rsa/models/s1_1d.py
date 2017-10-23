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
import edward as ed

def s1_1d(metaphor):
    vec_size,vec_kind = (25,'glove.twitter.27B.')
    subj,pred = metaphor

    quds = ['vicious','wet']
    possible_utterances = ['man','shark']

    real_vecs = load_vecs(mean=True,pca=True,vec_length=vec_size,vec_type=vec_kind)    
    inference_params = Inference_Params(
        vecs=real_vecs,
        subject=[subj],predicate=pred,
        quds=quds,
        possible_utterances=list(set(possible_utterances).union(set([pred]))),
        sig1=0.0005,sig2=0.005,
        qud_weight=0.0,freq_weight=0.0,
        categorical="categorical",
        sample_number = 500,
        number_of_qud_dimensions=1,
        burn_in=400,
        seed=False,trivial_qud_prior=False,
        step_size=0.0005,
        poss_utt_frequencies=defaultdict(lambda:1),
        qud_frequencies=defaultdict(lambda:1),
        qud_prior_weight=0.5,
        rationality=1.0,
        norm_vectors=False
        )

    run = Dist_RSA_Inference(inference_params)

    # run.compute_l1(load=0,save=False)

    run.compute_s1(s1_world=np.expand_dims(inference_params.vecs[subj],0),world_movement=False,debug=True)
    print(ed.get_session().run(run.s1_results))
    # import edward as ed
    # print(ed.get_session().run(run.s1_results))
    # results = run.qud_results()

    # run.compute_s1(params,s1_world=)

    # print("WORLD MOVEMENT\n:",run.world_movement("cosine",comparanda=[x for x in quds if x in real_vecs])[:50])
    # print("WORLD MOVEMENT WITH PROJECTION\n:",run.world_movement("cosine",comparanda=[x for x in quds if x in real_vecs],do_projection=True)[:50])
    # print("BASELINE:\n",sorted(qud_words,\
    #     key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[pred]],axis=0)),reverse=False)[:20])

    # print("RESULTS\n",[(x,np.exp(y)) for (x,y) in results[:20]])

    # return results

if __name__ == "__main__":
    s1_1d(("woman","rose"))
    # s1_1d(("bed","heaven"))
