from __future__ import division
import scipy
from utils.load_data import *
from utils.helperfunctions import *
from utils.helperfunctions import visualize_cosine
import numpy as np
import pickle
import itertools
from utils.refine_vectors import h_dict,processVecMatrix
import nltk
import glob
import os
from dbm import *
import pickle
from utils.load_adj_nouns import load_adj_nouns
from utils.load_data import get_words
# from lm_1b_eval import predict
from collections import defaultdict


vecs = pickle.load(open("data/word_vectors/glove.6B.mean_vecs300",'rb'),encoding='latin1')
# from utils.load_data import get_words
# frequencies,nouns,adjectives,verbs = get_words(preprocess=False)
# adjectives = list(set(open("data/adjectives.txt","r").read().split("\n")))
# nouns,adjectives,verbs = sorted(nouns, key=lambda x: frequencies[x], reverse=True),sorted(adjectives, key=lambda x: frequencies[x], reverse=True),sorted(verbs, key=lambda x: frequencies[x], reverse=True)
nouns,adjs = get_words_2()

dim=2
# for s2_qud in [['evil','ruthless'],['stubborn','strong']]:
for s2_qud in [['evil','cunning'],['stable','huge'],['fertile','boring']]:
    for vec_size,vec_kind in [(300,'glove.6B.')]:
# ,(200,'glove.twitter.27B.')]:
# ,(50,'glove.6B.'),(300,'glove.6B.'),(300,'glove.840B.')]:
        #         ["the", subj, "is", "a"]
        subj = 'man'

        abstract_threshold = 2.5
        print('abstract_threshold',abstract_threshold)
        concrete_threshold = 4.0
        print('concrete_threshold',concrete_threshold)

        qud_words = [a for a in list(adjs) if adjs[a] < abstract_threshold and a in vecs]

        # print("USING ALL ADJS")
        quds = list(list(zip(*visualize_cosine(vecs[subj],qud_words,vecs)))[0])
        # print(quds)
        # break
        quds=s2_qud
        # quds = list(set(quds[:50]).union(set(s2_qud)))

        possible_utterances = sorted([n for n in list(nouns) if nouns[n] > concrete_threshold and n in vecs],key=lambda x:scipy.spatial.distance.cosine(vecs[subj],vecs[x]),reverse=False)[:20:10]
        # ['devil','ox']
        # possible_utterances = ['field','devid','weasel','pine','postman','soldier','oak','scientist','devil','serpent'] 
        # +['chestnut']
        # print("BASELINE",quds[:20])
        print("QUDS:\n",quds[:200])
        print("UTTERANCES:\n",possible_utterances[:200])
        
        real_vecs = pickle.load(open("data/word_vectors/"+vec_kind+"pca and mean"+str(vec_size),'rb'),encoding='latin1')

        run = DistRSAInference(
        subject=[subj],predicate=possible_utterances[0],
        # possible_utterances=animals,
        # quds=animal_features,
        quds=quds,
        # quds = animal_features,
    #         ['unyielding']+list(list(zip(*visualize_cosine(np.mean([vecs['man'],vecs[word]],axis=0),freq_sorted_adjs,vecs)[:500:10]))[0]),

        possible_utterances=possible_utterances,
        # possible_utterances=
        # [noun for noun in nouns if noun not in adjectives][:100]+[adj for adj in adjectives if adj not in nouns][:100]+[pred],
        # sorted_nouns[sorted_nouns.index(pred) if pred in sorted_nouns else 500]+['horse'],
        object_name="animals_spec",
        mean_vecs=True,
        pca_remove_top_dims=True,
        sig1=0.0005,sig2=0.005,
    #         proposed_sig2*scaling_factor,
        qud_weight=0.0,freq_weight=0.0,
        categorical="categorical",
        vec_length=vec_size,vec_type=vec_kind,
        sample_number = 100,
        number_of_qud_dimensions=dim,
        burn_in=80,
        seed=False,trivial_qud_prior=False,
        step_size=0.0005,
        frequencies=defaultdict(lambda:1),
        qud_frequencies=defaultdict(lambda:1),
        qud_prior_weight=0.5,
        rationality=1.0,
        run_s2=True,
        speaker_world=s2_qud,
        s1_only=False,
        norm_vectors=False
        )
        # print(real_vecs[subj],real_vecs[pred])

        run.compute_results(load=0,save=False)
        # print(run.world_movement("cosine",do_projection=True,comparanda=[x for x in abstract_adjs+abstract_nouns if x in real_vecs]))
        # print("QUDS:\n",run.qud_results()[:20])
        # print("WORLD MOVEMENT\n:",run.world_movement("cosine",comparanda=[x for x in quds if x in real_vecs])[:50])
        # print("WORLD MOVEMENT\n:",run.world_movement("euclidean",comparanda=[x for x in quds if x in real_vecs])[:50])
        # print("BASELINE:\n:",run.baseline_model('mean')[:20])

        # raise Exception


