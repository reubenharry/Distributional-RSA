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
frequencies,nouns,adjectives,verbs = get_words(preprocess=False)
# adjectives = list(set(open("data/adjectives.txt","r").read().split("\n")))
nouns,adjectives,verbs = sorted(nouns, key=lambda x: frequencies[x], reverse=True),sorted(adjectives, key=lambda x: frequencies[x], reverse=True),sorted(verbs, key=lambda x: frequencies[x], reverse=True)
noun_rank,adj_rank = get_words_2()

# disjoined_nouns,disjoined_adjectives = sorted(disjoined_nouns, key=lambda x: frequencies[x], reverse=True),sorted(disjoined_adjectives, key=lambda x: frequencies[x], reverse=True)


def identification():

    vec_size=25
    vec_kind='glove.twitter.27B.'

    scores = []
    record = []
    # for pairs in ([('beautiful','scenery')],[('fiery','temper')]):
    for j,pairs in enumerate(zip(load_adj_nouns(real=False)[21:41],load_adj_nouns(real=True)[21:41])):
    # for pairs in ([('fiery','temper')],[('fiery','oven')]):
        print(j)
        for i,pair in enumerate(pairs):
            print(pair[0],pair[1])
            if pair[0] in vecs and pair[0] in adjectives and pair[1] in vecs:
                print("pair is",pair)


                pred = pair[0]
                subj = pair[1]
        #         ["the", subj, "is", "a"]

    #             distance  = scipy.spatial.distance.cosine(vecs[subj],vecs[pred])
    #             print(distance*scaling_factor)
                #prob_dict = predict("every")



        #         quds = list(list(zip(*visualize_cosine(np.mean([vecs[subj],vecs[pred]],axis=0),freq_sorted_adjs,vecs)))[0])
        #         quds = quds[:2*sorted_nouns.index(pred):2]
    #             quds = ["evil","leggy","cool","plotting","small"] + list(list(zip(*visualize_cosine(np.mean([vecs[subj],vecs[pred]],axis=0),freq_sorted_adjs,vecs)))[0])[:10]
        #         quds=quds[:quds.index('stable')]+['strong','stable','wooden','connected','leafy','unyielding','bearded']
                qud_words = [a for a in list(nouns) if a in vecs and a in noun_rank and noun_rank[a] > 2.0]
                # print("USING ALL ADJS")
                quds = list(list(zip(*visualize_cosine(np.mean([vecs[subj]],axis=0),qud_words,vecs)))[0])
                # print(quds)
                # break
                quds = quds[:2]
                if subj in quds:
                    quds.remove(subj)

                # possible_utterances = [a for a in list(adjectives) if a in vecs and a in adj_rank and adj_rank[a] < 3.0][:2]
                possible_utterances = [n for n in list(adjectives) if n in adj_rank and adj_rank[n] < 2.5 and n in vecs]
                possible_utterances = sorted(possible_utterances, key=lambda x:scipy.spatial.distance.cosine(vecs[subj],vecs[x]),reverse=False)[:2000:20]

                # sorted([n for n in list(adjs) if adjs[n] > 0.0 and n in vecs],key=lambda x:scipy.spatial.distance.cosine(vecs[pred],vecs[x]),reverse=False)[:1000:10]
                # +['chestnut']
                # print("BASELINE",quds[:20])
                print("QUDS:\n",quds[:20])
                print("UTTERANCES:\n",possible_utterances[:20])

                iden = DistRSAInference(
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
                sig1=0.0005,sig2=0.005,
            #         proposed_sig2*scaling_factor,
                qud_weight=0.0,freq_weight=0.0,
                categorical="categorical",
                vec_length=vec_size,vec_type=vec_kind,
                sample_number = 200,
                number_of_qud_dimensions=1,
                burn_in=100,
                seed=False,trivial_qud_prior=True,
                step_size=0.0005,
                frequencies=defaultdict(lambda:1),
                qud_frequencies=defaultdict(lambda:1),
                qud_prior_weight=0.5,
                rationality=0.7,
                run_s2=False,
                speaker_world=vecs[subj],
                s1_only=False,
                norm_vectors=True
                )
                # real_vecs = pickle.load(open("data/word_vectors/"+vec_kind+"pca and mean"+str(vec_size),'rb'),encoding='latin1')
                # print(real_vecs[subj],real_vecs[pred])

                iden.compute_results(load=0,save=False)
                # print(run.world_movement("cosine",do_projection=True,comparanda=[x for x in abstract_adjs+abstract_nouns if x in real_vecs]))
                # print("WORLD MOVEMENT\n:",run.world_movement("cosine",comparanda=[x for x in quds if x in real_vecs])[:50])
                # print("WORLD MOVEMENT\n:",run.world_movement("euclidean",comparanda=[x for x in quds if x in real_vecs])[:50])
                # print("BASELINE:\n:",run.baseline_model('mean')[:20])

                # raise Exception
                result = iden.qud_results()
                print("gold label:",bool(i),"|prediction: ",result)

                scores.append((bool(i),result))
                correct_scores = [x for x in scores if x[0]==x[1]]
                accuracy = len(correct_scores) / len(scores)
                record.append((pair,bool(i),result))
                print("accuracy:",accuracy)

for i in range(1):
    identification()

