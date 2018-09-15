from __future__ import division

# log_path = input("logging path: ")

from collections import defaultdict
import scipy
import numpy as np
import pickle
import itertools
from dist_rsa.dbm import *
from dist_rsa.utils.load_data import *
from dist_rsa.utils.helperfunctions import *
from dist_rsa.utils.config import abstract_threshold,concrete_threshold

import tensorflow as tf

word_selection_vecs = load_vecs(mean=False,pca=False,vec_length=300,vec_type='glove.6B.')

def l1_model(subj,pred,hyperparams):
    vec_size,vec_kind = 300,'glove.6B.'
    freqs = pickle.load(open('dist_rsa/data/google_freqs/freqs','rb'))

    possible_utterances, quds = get_possible_utterances_and_quds(subj="voice",pred="river",word_selection_vecs=word_selection_vecs)
    possible_utterances = sorted(possible_utterances[:50])
    # possible_utterances = sorted(possible_utterances[:5])
    # quds = sorted(list(set(quds[:50])))


    quds = ["smaller","bigger"]
    # possible_utterances = ["horse","man"]

    print("QUDS",quds[:10]) 
    print("UTTERANCES:\n",possible_utterances[:10])

    vecs = load_vecs(mean=hyperparams.mean_center,pca=hyperparams.remove_top_dims,vec_length=vec_size,vec_type=vec_kind) 
    for x in possible_utterances:
        if x not in vecs:
            # print(x,"not in vecs")
            possible_utterances.remove(x)

    params = Inference_Params(
        vecs=vecs,
        subject=[subj],predicate=pred,
        quds=sorted(quds),
        possible_utterances=sorted(list(set(possible_utterances).union(set([pred])))),
        sig1=hyperparams.sig1,sig2=hyperparams.sig2,l1_sig1=hyperparams.l1_sig1,
        qud_weight=0.0,freq_weight=0.0,
        number_of_qud_dimensions=1,
        poss_utt_frequencies=defaultdict(lambda:1),
        qud_frequencies=defaultdict(lambda:1),
        rationality=1.0,
        norm_vectors=hyperparams.norm_vectors,
        heatmap=False,
        resolution=Resolution(span=10,number=100),
        model_type="numpy_discrete_mixture",
        # model_type="discrete_mixture",
        calculate_projected_marginal_world_posterior=True,
        )

    run = Dist_RSA_Inference(params)
    run.compute_l1(load=0,save=False)

    print("marginal",params.qud_marginals)

    out = run.tf_results
    del run
    del vecs
    tf.reset_default_graph()
    return params


if __name__ == "__main__":

    subj = "banana"
    pred = "river"

    mean_center = True
    remove_top_dims = False
    norm_vectors = True
    sig1 = 1.0
    sig2 = 1e-2
    l1_sig1 = 10.0
    hyperparams = Hyperparams(mean_center=mean_center,remove_top_dims=remove_top_dims,norm_vectors=norm_vectors,sig1=sig1,sig2=sig2,l1_sig1=l1_sig1)
    print("HYPERPARAMS",hyperparams.show())

    results = l1_model(subj=subj,pred=pred,hyperparams=hyperparams)





