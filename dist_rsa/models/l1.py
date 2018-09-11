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

# vec_size,vec_kind = 25,'glove.twitter.27B.'
word_selection_vecs = load_vecs(mean=False,pca=False,vec_length=300,vec_type='glove.6B.')

def l1_model(subj,pred,hyperparams):
    vec_size,vec_kind = 300,'glove.6B.'
    freqs = pickle.load(open('dist_rsa/data/google_freqs/freqs','rb'))


    # get_possible_utterance
    # get_quds

    # word_selection_vecs = load_vecs(mean=False,pca=False,vec_length=vec_size,vec_type=vec_kind)
    # nouns,adjs = get_words(with_freqs=False)
    # print('abstract_threshold',abstract_threshold)
    # print('concrete_threshold',concrete_threshold)
    # qud_words = [a for a in list(adjs) if adjs[a] < abstract_threshold and a in word_selection_vecs]

    # quds = sorted(qud_words,\
    #     key=lambda x:scipy.spatial.distance.cosine(word_selection_vecs[x],np.mean([word_selection_vecs[pred],word_selection_vecs[subj]],axis=0)),reverse=False)
    #     # key=lambda x:freqs[x],reverse=True)

    # noun_words = [n for n in nouns if nouns[n] > concrete_threshold and n in word_selection_vecs]
    # possible_utterances = sorted(noun_words,\
    #     key=lambda x: scipy.spatial.distance.cosine(word_selection_vecs[x],np.mean([word_selection_vecs[subj],word_selection_vecs[subj]],axis=0)),reverse=False)
    #     # key=lambda x:freqs[x],reverse=True)


            # raise Exception("utterance not in word_selection_vecs")
    possible_utterances, quds = get_possible_utterances_and_quds(subj=subj,pred=pred,word_selection_vecs=word_selection_vecs)
    possible_utterances = sorted(possible_utterances[:200])
    quds = sorted(list(set(quds[:100])))

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
        possible_utterances=list(set(possible_utterances).union(set([pred]))),
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
        calculate_projected_marginal_world_posterior=True,
        )

    run = Dist_RSA_Inference(params)
    run.compute_l1(load=0,save=False)

    print("marginal",params.qud_marginals[0])

    out = run.tf_results
    del run
    del vecs
    tf.reset_default_graph()
    return params

    # world_means = run.world_samples
    # print(world_means[:5],"MEANS")

    # print(results[:5])

    # if not is_baseline:
    #     worldm = run.world_movement("cosine",comparanda=[x for x in qud_words if x in vecs])
    #     # print("\nworld\n",worldm[:5])
    # else: worldm = None
        # out.write("\nWORLD MOVEMENT:\n")
        # out.write(str(worldm))
    # print("WORLD MOVEMENT WITH PROJECTION\n:",run.world_movement("cosine",comparanda=[x for x in quds if x in vecs],do_projection=True)[:50])
    # print("BASELINE:\n",sorted(qud_words,\
    #     key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[pred]],axis=0)),reverse=False)[:5])

    # demarg = demarginalize_product_space(results)
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

if __name__ == "__main__":

    pass





