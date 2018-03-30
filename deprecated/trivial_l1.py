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
from csv_munger import *
from collections import defaultdict


vecs = pickle.load(open("data/word_vectors/glove.6B.mean_vecs300",'rb'),encoding='latin1')
# from utils.load_data import get_words
# frequencies,nouns,adjectives,verbs = get_words(preprocess=False)
# adjectives = list(set(open("data/adjectives.txt","r").read().split("\n")))
# nouns,adjectives,verbs = sorted(nouns, key=lambda x: frequencies[x], reverse=True),sorted(adjectives, key=lambda x: frequencies[x], reverse=True),sorted(verbs, key=lambda x: frequencies[x], reverse=True)
nouns,adjs = get_words_2()

# disjoined_nouns,disjoined_adjectives = sorted(disjoined_nouns, key=lambda x: frequencies[x], reverse=True),sorted(disjoined_adjectives, key=lambda x: frequencies[x], reverse=True)

for vec_size,vec_kind in [(25,'glove.twitter.27B.')]:
# ,(200,'glove.twitter.27B.')]:
# ,(50,'glove.6B.'),(300,'glove.6B.'),(300,'glove.840B.')]:
    for j in range(1):
        # ('man','lion'),('woman','lion'),('man','sheep'),('man','tree'),('woman','tree'),
        # ('man','lion'),('woman','lion'),
        for subj,pred in [('room','oven'),('leader','butcher'),('butcher','leader'),('man','sheep'),('love','prison'),('time',"illusion")]:
    # for pred in ['lion','poison']:
    #     for subj in ["love"]:
            # prob_dict = predict(" ".join([subj, "is","a"]))
            # filtered_nouns = [x for x in nouns if x in prob_dict and x not in adjectives and x not in verbs]
            # sorted_nouns = sorted(filtered_nouns,key=lambda x : prob_dict[x],reverse=True)
            # filtered_adjs = [x for x in adjectives if x in prob_dict and x not in nouns and x not in verbs]
            # sorted_adjs = sorted(filtered_adjs,key=lambda x : prob_dict[x],reverse=True)
            # for sig1 in [5e-3,5e-4]:
            #     for sig2 in [5e-2,5e-3]:
            # qud_frequencies = {}
            # for word in [x for x in nouns+adjectives if x in vecs]:
            #     qud_frequencies[word] = scipy.spatial.distance.cosine(np.mean([vecs[pred],vecs[subj]],axis=0),vecs[word])

            # prob_dict = predict(" ".join([subj, "is","a"]))
            # filtered_disjoined_nouns = [x for x in disjoined_nouns if x in prob_dict and x not in disjoined_adjectives and x in vecs]
            # sorted_disjoined_nouns = sorted(filtered_disjoined_nouns,key=lambda x : prob_dict[x],reverse=True)
            for i in range(3):
        #         ["the", subj, "is", "a"]

    #             distance  = scipy.spatial.distance.cosine(vecs[subj],vecs[pred])
    #             print(distance*scaling_factor)
                #prob_dict = predict("every")



        #         quds = list(list(zip(*visualize_cosine(np.mean([vecs[subj],vecs[pred]],axis=0),freq_sorted_adjs,vecs)))[0])
        #         quds = quds[:2*sorted_nouns.index(pred):2]
    #             quds = ["evil","leggy","cool","plotting","small"] + list(list(zip(*visualize_cosine(np.mean([vecs[subj],vecs[pred]],axis=0),freq_sorted_adjs,vecs)))[0])[:10]
        #         quds=quds[:quds.index('stable')]+['strong','stable','wooden','connected','leafy','unyielding','bearded']
                qud_words = [a for a in list(adjs) if adjs[a] < 3.5 and a in vecs]
                print("USING ALL ADJS")
                quds = list(list(zip(*visualize_cosine(np.mean([vecs[pred],vecs[subj]],axis=0),qud_words,vecs)))[0])
                # print(quds)
                # break
                quds = quds[:100]

                possible_utterances = sorted([n for n in list(nouns) if nouns[n] > 3.5 and n in vecs],key=lambda x:scipy.spatial.distance.cosine(vecs[pred],vecs[x]),reverse=False)[:2000:10]
                # +['chestnut']
                # print("BASELINE",quds[:20])
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
                sample_number = 100,
                number_of_qud_dimensions=2,
                burn_in=85,
                seed=False,trivial_qud_prior=False,
                step_size=0.0005,
                frequencies=defaultdict(lambda:1),
                qud_frequencies=defaultdict(lambda:1),
                qud_prior_weight=0.5,
                rationality=0.7,
                run_s2=False,
                speaker_world=vecs[subj],
                s1_only=True
                )
                real_vecs = pickle.load(open("data/word_vectors/"+vec_kind+"pca and mean"+str(vec_size),'rb'),encoding='latin1')
                # print(real_vecs[subj],real_vecs[pred])

                run.compute_results(load=0,save=False)
                # print(run.world_movement("cosine",do_projection=True,comparanda=[x for x in abstract_adjs+abstract_nouns if x in real_vecs]))
                print("QUDS:\n",run.qud_results()[:20])
                print("WORLD MOVEMENT\n:",run.world_movement("cosine",comparanda=[x for x in quds if x in real_vecs])[:50])
                # print("WORLD MOVEMENT\n:",run.world_movement("euclidean",comparanda=[x for x in quds if x in real_vecs])[:50])
                print("BASELINE:\n:",run.baseline_model('mean')[:20])

                # raise Exception


