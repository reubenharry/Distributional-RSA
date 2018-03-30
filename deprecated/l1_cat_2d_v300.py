from __future__ import division
import scipy
import numpy as np
import pickle
import itertools
import nltk
import glob
import os
from collections import defaultdict

from dbm import *
from dist_rsa.utils.load_data import *
from dist_rsa.utils.helperfunctions import *
from dist_rsa.utils.load_data import get_freqs

def l1_cat_2d_v300(metaphor):

    output_dict = {}

    vecs = pickle.load(open("dist_rsa/data/word_vectors/glove.6B.mean_vecs300",'rb'))
    # from utils.load_data import get_words
    # frequencies,nouns,adjectives,verbs = get_words(preprocess=False)
    # adjectives = list(set(open("data/adjectives.txt","r").read().split("\n")))
    # nouns,adjectives,verbs = sorted(nouns, key=lambda x: frequencies[x], reverse=True),sorted(adjectives, key=lambda x: frequencies[x], reverse=True),sorted(verbs, key=lambda x: frequencies[x], reverse=True)
    nouns,adjs = get_words()

    vec_size,vec_kind = 300,'glove.6B.'
        
    subj,pred = metaphor
    abstract_threshold = 2.5
    print('abstract_threshold',abstract_threshold)
    concrete_threshold = 3.0
    print('concrete_threshold',concrete_threshold)


    sig2_distance = scipy.spatial.distance.cosine(vecs[subj],vecs[pred])
    print("SUBJ PRED DISTANCE",sig2_distance,"\nproposed scaling factor:",0.005/sig2_distance)
    scaling_factor = 0.01
    print("SCALING FACTOR",scaling_factor)

    qud_words = [a for a in list(adjs) if adjs[a] < abstract_threshold and a in vecs]
    quds = sorted(qud_words,\
        key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[subj]],axis=0)),reverse=False)
    possible_utterances = sorted([n for n in list(nouns) if nouns[n] > concrete_threshold and n in vecs],\
        key=lambda x:scipy.spatial.distance.cosine(vecs[x],vecs[subj]),reverse=False)
    # print(quds)
    # break
    quds = quds[:100]
    possible_utterances = possible_utterances[:1000:10]

    # +['chestnut']
    # print("BASELINE",quds[:20])
    print("QUDS:\n",quds[:200])
    print("UTTERANCES:\n",possible_utterances[:20])

    real_vecs = pickle.load(open("dist_rsa/data/word_vectors/"+vec_kind+"pca and mean"+str(vec_size),'rb'),encoding='latin1')


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
    sig1=0.0005,sig2=float(sig2_distance*scaling_factor),
#         proposed_sig2*scaling_factor,
    qud_weight=0.0,freq_weight=0.0,
    categorical="categorical",
    vec_length=vec_size,vec_type=vec_kind,
    sample_number = 300,
    number_of_qud_dimensions=2,
    burn_in=200,
    seed=False,trivial_qud_prior=False,
    step_size=0.0005,
    frequencies=defaultdict(lambda:1),
    qud_frequencies=defaultdict(lambda:1),
    qud_prior_weight=0.5,
    rationality=1.0,
    run_s2=False,
    speaker_world=real_vecs[subj],
    s1_only=False,
    display_l1_world_movement=False

    )

    run.compute_results(load=0,save=False)
    # print(run.world_movement("cosine",do_projection=True,comparanda=[x for x in abstract_adjs+abstract_nouns if x in real_vecs]))
    results = run.qud_results()
    print("QUDS:\n",results[:20])
    print("WORLD MOVEMENT\n:",run.world_movement("cosine",comparanda=[x for x in quds if x in real_vecs])[:50])
    # print("WORLD MOVEMENT\n:",run.world_movement("euclidean",comparanda=[x for x in quds if x in real_vecs])[:50])
    print("BASELINE:\n:",run.baseline_model('mean')[:20])

    return results

if __name__ == "__main__":
    l1_cat_2d_v300(('man','lion'))
