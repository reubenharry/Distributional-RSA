from collections import defaultdict
import scipy
import numpy as np
import pickle
import itertools
from dist_rsa.dbm import *
from dist_rsa.utils.helperfunctions import *
from dist_rsa.utils.config import abstract_threshold,concrete_threshold
import edward as ed
from dist_rsa.utils.simple_vecs import real_vecs as simple_vecs
import itertools




def l1_model(subj,pred,sig1,sig2,l1_sig1,quds,possible_utterances):
    # vec_size,vec_kind = 25,'glove.twitter.27B.'

    vecs = simple_vecs
    real_vecs= simple_vecs

    print("UTTERANCES:\n",sorted(list(set(possible_utterances).union(set([pred]))))[:20])

    params = Inference_Params(
        vecs=real_vecs,
        subject=[subj],predicate=pred,
        quds=quds,
        possible_utterances=sorted(list(set(possible_utterances).union(set([pred])))),
        sig1=sig1,sig2=sig2, l1_sig1=l1_sig1,
        qud_weight=0.0,freq_weight=0.0,
        number_of_qud_dimensions=1,
        poss_utt_frequencies=defaultdict(lambda:1),
        qud_frequencies=defaultdict(lambda:1),
        rationality=1.0,
        norm_vectors=False,
        resolution=Resolution(span=10,number=100),
        # model_type="discrete_mixture",
        # model_type="discrete_exact",
        model_type="numpy_discrete_mixture",
        heatmap=False,
        calculate_projected_marginal_world_posterior=True,
        )

    run = Dist_RSA_Inference(params)
    run.compute_l1(load=0,save=False)
    tf_results = run.tf_results

    print(params.marginal_means)

    return tf_results

if __name__ == "__main__":

    # heatmaps, quds = l1_model(subj="child",pred="shark",sig1=5.0,sig2=0.5,l1_sig1=5.0,quds=["vicious","wonder"],possible_utterances=["shark","swimmer","man"])
    heatmaps, quds = l1_model(subj="man",pred="swimmer",sig1=0.1,sig2=0.1,l1_sig1=0.1,quds=["swims","vicious"],possible_utterances=["shark","swimmer","man"])
    print(quds)



