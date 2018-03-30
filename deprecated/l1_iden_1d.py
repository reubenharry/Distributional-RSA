from __future__ import division
import scipy
import numpy as np
import itertools
import nltk
import glob
import os
import pickle
from collections import defaultdict
from dist_rsa.dbm import *
from dist_rsa.utils.load_data import *
from dist_rsa.utils.load_data import get_words
from dist_rsa.utils.helperfunctions import *
from dist_rsa.lm_1b_eval import predict



def l1_iden_1d(metaphor):

    vecs = pickle.load(open("dist_rsa/data/word_vectors/glove.6B.mean_vecs300",'rb'),encoding='latin1')
    nouns,adjs = get_words()

    vec_size,vec_kind = (25,'glove.twitter.27B.')

    subj,pred = metaphor
    abstract_threshold = 2.5
    print('abstract_threshold',abstract_threshold)
    concrete_threshold = 3.0
    print('concrete_threshold',concrete_threshold)
                
    qud_words = [n for n in nouns if nouns[n] < 4.0 and n in vecs]
    qud_words = sorted(qud_words,\
        key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[pred]],axis=0)),reverse=False)
    # print(quds)
    # break
    quds = qud_words[:30]

    # prob_dict = predict(" ".join([subj, "are"]))
    possible_utterance_nouns = sorted([a for a in adjs if adjs[a] > concrete_threshold and a in vecs],\
        # key=lambda x:prob_dict[x],reverse=True)
        key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[pred]],axis=0)),reverse=False)

    possible_utterances = possible_utterance_nouns[:30]

    print("QUDS:\n",quds[:20])
    print("UTTERANCES:\n",possible_utterances[:20])

    run = DistRSAInference(
    subject=[subj],predicate=pred,
    # possible_utterances=animals,
    # quds=animal_features,
    quds=quds,
    # quds = animal_features,
#         ['unyielding']+list(list(zip(*visualize_cosine(np.mean([vecs['man'],vecs[word]],axis=0),freq_sorted_adjs,vecs)[:500:10]))[0]),

    possible_utterances=list(set(possible_utterances).union(set([pred]))),
    # possible_utterances=
    # [noun for noun in nouns if noun not in adjectives][:100]+[adj for adj in adjectives if adj not in nouns][:100]+[pred],
    # sorted_nouns[sorted_nouns.index(pred) if pred in sorted_nouns else 500]+['horse'],
    object_name="animals_spec",
    mean_vecs=True,
    pca_remove_top_dims=True,
    sig1=0.0005,sig2=0.01,
#         proposed_sig2*scaling_factor,
    qud_weight=0.0,freq_weight=0.0,
    categorical="categorical",
    vec_length=vec_size,vec_type=vec_kind,
    sample_number = 400,
    number_of_qud_dimensions=1,
    burn_in=285,
    seed=False,trivial_qud_prior=True,
    step_size=0.0005,
    frequencies=defaultdict(lambda:1),
    qud_frequencies=defaultdict(lambda:1),
    qud_prior_weight=0.9,
    rationality=1.0,
    run_s2=False,
    speaker_world=vecs[subj],
    s1_only=False,
    norm_vectors=True
    )
    real_vecs = pickle.load(open("dist_rsa/data/word_vectors/"+vec_kind+"pca and mean"+str(vec_size),'rb'),encoding='latin1')
    # print(real_vecs[subj],real_vecs[pred])

    run.compute_results(load=0,save=False)
    # print(run.world_movement("cosine",do_projection=True,comparanda=[x for x in abstract_adjs+abstract_nouns if x in real_vecs]))
    print(run.qud_results())
    # print("QUDS:\n",results[:20])
    # print("WORLD MOVEMENT\n:",run.world_movement("cosine",comparanda=[x for x in quds if x in real_vecs])[:50])
    # print("WORLD MOVEMENT\n:",run.world_movement("euclidean",comparanda=[x for x in quds if x in real_vecs])[:50])
    # print("BASELINE:\n:",run.baseline_model('mean')[:20])


    return run.qud_samples
                    # raise Exception

if __name__ == "__main__":

    l1_iden_1d(('man','lion'))
    l1_iden_1d(('animal','lion'))
    l1_iden_1d(('man','postman'))

