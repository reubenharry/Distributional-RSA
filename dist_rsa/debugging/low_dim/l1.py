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
import itertools




def l1_model(subj,pred,sig1,sig2,l1_sig1,mixture_variational,resolution,quds,only_trivial,just_s1,just_l0,possible_utterances,discrete,variational,step_size):
    vec_size,vec_kind = 25,'glove.twitter.27B.'
    vecs = simple_vecs
    real_vecs= simple_vecs

    for x in possible_utterances:
        if x not in real_vecs:
            print(x,"not in vecs")
            possible_utterances.remove(x)
            # raise Exception("utterance not in vecs")

    print("UTTERANCES:\n",sorted(list(set(possible_utterances).union(set([pred]))))[:20])


    params = Inference_Params(
        vecs=real_vecs,
        subject=[subj],predicate=pred,
        quds=quds,
        possible_utterances=sorted(list(set(possible_utterances).union(set([pred])))),
        sig1=sig1,sig2=sig2, l1_sig1=l1_sig1,
        qud_weight=0.0,freq_weight=0.0,
        categorical="categorical",
        sample_number = 50,
        number_of_qud_dimensions=1,
        burn_in=1000,
        seed=False,trivial_qud_prior=False,
        step_size=step_size,
        poss_utt_frequencies=defaultdict(lambda:1),
        qud_frequencies=defaultdict(lambda:1),
        qud_prior_weight=0.5,
        rationality=1.0,
        norm_vectors=False,
        variational=variational,
        variational_steps=1000,
        baseline=False,
        discrete_l1=discrete,
        resolution=resolution,
        only_trivial=only_trivial,
        just_s1=just_s1,
        just_l0=just_l0,
        target_qud=0,
        mixture_variational=mixture_variational,
        )

    run = Dist_RSA_Inference(params)

    run.compute_l1(load=0,save=False)



    # tf_results = run.tf_results
    return run

if __name__ == "__main__":

    quds=['vicious','swims']
    poss_utts=["shark","swimmer"]
    # b = l1_model(subj="man",pred="swimmer",sig1=1.0,sig2=0.1,l1_sig1=1.0,resolution=(200,0.01),quds=["vicious"],possible_utterances=poss_utts, only_trivial=False,just_s1=False,just_l0=False,discrete=True,variational=False,step_size=1e-10,mixture_variational=False)
    a = l1_model(subj="man",pred="swimmer",sig1=1.0,sig2=0.1,l1_sig1=1.0,resolution=(200,0.01),quds=["vicious"],possible_utterances=poss_utts, only_trivial=False,just_s1=False,just_l0=False,discrete=False,variational=True,step_size=1e-10,mixture_variational=True)

    

    print("\n\nRESULTS:\n")
    print("")
    print("mixture:",[(x,np.exp(y)) for (x,y) in a.tf_results[1]]) 
    print("discrete:",[(x,np.exp(y)) for (x,y) in b.tf_results[1]])

