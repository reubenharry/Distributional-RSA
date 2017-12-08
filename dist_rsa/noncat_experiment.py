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
#to try: vary dimensions, number of utts, learning rate, large categorical with quds to see if poss utts are good, 
#other animals, large qud set
#iden checks: switch num of verbs and adjs


#EXPERIMENTS:
#Is noncat stable enough on animals?
#Is fast cat stable enough in general?
#Is identification workable with smaller sample size?
#

#YES GOOD IDEA: do experiments to see if bigrams etc help with animals

#KEY: what counts as a good QUD? not clear i have a good sense of this: use might be better:
    #e.g. which metaphors convey a similar thing about something: already have a metric for this


# cat = DistRSAInference(
#     subject=['man'],predicate=animal,
#     quds=animal_features,possible_utterances=animal+nouns[50:1000],
#     object_name="animals_non_cat",
#     mean_vecs=True,
#     pca_remove_top_dims=False,
#     sig1=10.0,sig2=0.05,
#     qud_weight=0.0,freq_weight=1.0,
#     categorical="categorical",
#     vec_length=50,vec_type="glove.6B.",
#     sample_number = 50000,
#     number_of_qud_dimensions=1,
#     burn_in=25000,
#     seed=False,trivial_qud_prior=False,
#     )

# noncat = DistRSAInference(
#     subject=['man'],predicate="sheep",
#     # quds=adjectives[:1000]+['predatory'],possible_utterances=nouns[:1000]+['shark'],
#     quds=animal_features,possible_utterances=nouns[50:1000]+animals,
#     object_name="animals_non_cat",
#     mean_vecs=True,
#     pca_remove_top_dims=False,
#     sig1=10.0,sig2=0.05,
#     qud_weight=0.0,freq_weight=1.0,
#     categorical="non-categorical",
#     vec_length=50,vec_type="glove.6B.",
#     sample_number = 50000,
#     number_of_qud_dimensions=1,
#     burn_in=25000,
#     seed=False,trivial_qud_prior=False,
#     )

# iden_1 = DistRSAInference(
#     subject=['crop'],predicate="wither",
#     # quds=adjectives[:1000]+['predatory'],possible_utterances=nouns[:1000]+['shark'],
#     quds=adjectives[10:500],possible_utterances=verbs[10:1000]+['wither'],
#     object_name="iden_crop",
#     mean_vecs=True,
#     pca_remove_top_dims=False,
#     sig1=1.0,sig2=1.0,
#     qud_weight=0.0,freq_weight=1.0,
#     categorical="categorical",
#     vec_length=50,vec_type="glove.6B.",
#     sample_number = 1000,
#     number_of_qud_dimensions=1,
#     burn_in=500,
#     seed=False,trivial_qud_prior=True,
#     )

# iden_2 = DistRSAInference(
#     subject=['hope'],predicate="wither",
#     # quds=adjectives[:1000]+['predatory'],possible_utterances=nouns[:1000]+['shark'],
#     quds=adjectives[10:500],possible_utterances=verbs[10:500]+['wither'],
#     object_name="iden_hope",
#     mean_vecs=True,
#     pca_remove_top_dims=False,
#     sig1=1.0,sig2=1.0,
#     qud_weight=0.0,freq_weight=0.0,
#     categorical="categorical",
#     vec_length=50,vec_type="glove.6B.",
#     sample_number = 1000,
#     number_of_qud_dimensions=1,
#     burn_in=500,
#     seed=False,trivial_qud_prior=True,
#     )

# import edward as ed
# ed.set_seed(42)
#
vecs = pickle.load(open("data/word_vectors/glove.6B.mean_vecs50",'rb'),encoding='latin1')

# model_dict = pickle.load(open("data/model_dict",'rb'))
# top_nouns = pickle.load(open("data/top_nouns",'rb'))

from utils.load_data import get_words

frequencies,nouns,adjectives,verbs = get_words(preprocess=False)

print("using better adjectives")
adjectives = list(set(open("data/adjectives.txt","r").read().split("\n")))

nouns,adjectives,verbs = sorted(nouns, key=lambda x: frequencies[x], reverse=True),sorted(adjectives, key=lambda x: frequencies[x], reverse=True),sorted(verbs, key=lambda x: frequencies[x], reverse=True)
from lm_1b_eval import predict

def identification():

    # prob_dict = predict("The man is a")

    # filtered_nouns = [x for x in nouns if x in prob_dict and x in vecs]
    # sorted_nouns = sorted(filtered_nouns,key=lambda x : prob_dict[x],reverse=True)

    # filtered_adjs = [x for x in adjectives if x in prob_dict and x in vecs]
    # sorted_adjs = sorted(filtered_adjs,key=lambda x : prob_dict[x],reverse=True)

    # freq_sorted_adjs = sorted(filtered_adjs,key=lambda x : frequencies[x],reverse=True)
    # freq_sorted_nouns = sorted(filtered_nouns,key=lambda x : frequencies[x],reverse=True)
    # vecs = pickle.load(open("data/word_vectors/"+"glove.6B.mean_vecs50",'rb'),encoding='latin1')
    # frequencies,_,_,_ = get_words(preprocess=True)
    scores = []
    record = []
    for j,pairs in enumerate(zip(load_adj_nouns(real=False)[21:41],load_adj_nouns(real=True)[21:41])):
        print(j)
        for i,pair in enumerate(pairs):
            if pair[0] in vecs and pair[0] in adjectives and pair[1] in vecs:
                print("pair is",pair)


                pred = pair[0]
                subj = pair[1]

                prob_dict = predict(" ".join([subj, "is"]))
                filtered_nouns = [x for x in nouns if x in prob_dict and x not in adjectives and x not in verbs]
                sorted_nouns = sorted(filtered_nouns,key=lambda x : prob_dict[x],reverse=True)
                filtered_adjs = [x for x in adjectives if x in prob_dict and x not in nouns and x not in verbs]
                sorted_adjs = sorted(filtered_adjs,key=lambda x : prob_dict[x],reverse=True)

            #     iden = DistRSAInference(
            #     subject=[pair[1]],predicate=pair[0],
            #     # possible_utterances=animals,
            #     # quds=animal_features,
            #     quds=list(set([]+list(list(zip(*visualize_cosine(np.mean([vecs[pair[0]],vecs[pair[1]]],axis=0),[x for x in adjectives if x in vecs],vecs)[:200]))[0]))),
            #     # quds = animal_features,
            # #         ['unyielding']+list(list(zip(*visualize_cosine(np.mean([vecs['man'],vecs[word]],axis=0),freq_sorted_adjs,vecs)[:500:10]))[0]),

            #     # possible_utterances=sorted_nouns[:1000]+[pred],
            #     possible_utterances=nouns[:500]+adjectives[:500]+[pair[0]],
            #     # sorted_nouns[sorted_nouns.index(pred) if pred in sorted_nouns else 500]+['horse'],
            #     object_name="animals_spec",
            #     mean_vecs=True,
            #     pca_remove_top_dims=False,
            #     sig1=0.001,sig2=0.1,
            # #         proposed_sig2*scaling_factor,
            #     qud_weight=0.0,freq_weight=1.0,
            #     categorical="categorical",
            #     vec_length=50,vec_type="glove.6B.",
            #     sample_number = 1000,
            #     number_of_qud_dimensions=1,
            #     burn_in=850,
            #     seed=False,trivial_qud_prior=True,
            #     step_size=0.0005,
            #     frequencies=frequencies,
            #     qud_prior_weight=0.5,
            #     rationality=0.99,
            #     run_s2=False,
            #     speaker_world=vecs['the']+(1/10*vecs["unyielding"]),
            #     s1_only=False
            #     )

                quds = list(list(zip(*visualize_cosine(np.mean([vecs[pred],vecs[subj]],axis=0),[x for x in adjectives if x in vecs],vecs)[:500]))[0])
                # print("BASELINE",quds[:20])

                run = DistRSAInference(
                subject=[subj],predicate=pred,
                # possible_utterances=animals,
                # quds=animal_features,
                quds=quds,
                # quds = animal_features,
            #         ['unyielding']+list(list(zip(*visualize_cosine(np.mean([vecs['man'],vecs[word]],axis=0),freq_sorted_adjs,vecs)[:500:10]))[0]),

                possible_utterances=sorted_nouns[:2000]+sorted_adjs[:2000]+[pred],
                # possible_utterances=
                # [noun for noun in nouns if noun not in adjectives][:100]+[adj for adj in adjectives if adj not in nouns][:100]+[pred],
                # sorted_nouns[sorted_nouns.index(pred) if pred in sorted_nouns else 500]+['horse'],
                object_name="animals_spec",
                mean_vecs=True,
                pca_remove_top_dims=False,
                sig1=0.001,sig2=0.01,
            #         proposed_sig2*scaling_factor,
                qud_weight=1.0,freq_weight=1.0,
                categorical="categorical",
                vec_length=25,vec_type="glove.twitter.27B.",
                sample_number = 100,
                number_of_qud_dimensions=1,
                burn_in=90,
                seed=False,trivial_qud_prior=True,
                step_size=0.0005,
                frequencies=prob_dict,
                qud_prior_weight=0.5,
                rationality=1.0,
                run_s2=False,
                speaker_world=vecs[subj]+(1/10*vecs["unyielding"]),
                s1_only=False
                )
                
                iden.compute_results(load=0,save=False)

                # print(iden.world_movement("cosine"))

                print("gold label:",bool(i),"|prediction: ",iden.qud_results())

                scores.append((bool(i),iden.qud_results()))
                correct_scores = [x for x in scores if x[0]==x[1]]
                accuracy = len(correct_scores) / len(scores)
                record.append((pair,bool(i),iden.qud_results()))
                print("accuracy:",accuracy)
            else:
                print(pair)

    print(record)    

# identification()
# raise Exception




# noncat = DistRSAInference(
# subject=['man'],predicate="lion",
# quds=animal_features,possible_utterances=verbs[50:1000]+adjectives[50:1000]+nouns[50:1000]+['lion'],
# object_name="animals_non_cat",
# mean_vecs=True,
# pca_remove_top_dims=False,
# sig1=10.0,sig2=0.05,
# qud_weight=0.0,freq_weight=1.0,
# categorical="non-categorical",
# vec_length=50,vec_type="glove.6B.",
# sample_number = 10000,
# number_of_qud_dimensions=1,
# burn_in=5000,
# seed=False,trivial_qud_prior=False,
# step_size=0.04
# )

# noncat = DistRSAInference(
# subject=['man'],predicate="sheep",
# # quds=adjectives[:1000]+['predatory'],possible_utterances=nouns[:1000]+['shark'],
# quds=adjectives[50:1000],possible_utterances=nouns[50:100]+animals,
# object_name="animals_non_cat_multi_dim",
# mean_vecs=True,
# pca_remove_top_dims=False,
# sig1=10.0,sig2=0.04,
# qud_weight=0.0,freq_weight=1.0,
# categorical="non-categorical",
# vec_length=300,vec_type="glove.6B.",
# number_of_qud_dimensions=2,
# sample_number = 10000,
# burn_in=5000,
# seed=False,trivial_qud_prior=False,
# step_size=0.01
# )
scaling_factor = 1e1

# print("baseline:",list(list(zip(*visualize_cosine(np.mean([vecs['man'],vecs['tree']],axis=0),sorted_adjs,vecs)[:500]))[0]))
def run():
    # for word in ["tree","shark","sheep","lion","fox","rabbit","ox","weasel","cow"]:
    for word in ['oak']:
        for i in range(1,2):
            # for subj in ["life", "man","sea"]:
            for subj in ['man','tree']:

                subject = [subj]
        
                distance  = scipy.spatial.distance.cosine(vecs[subj],vecs[word])
                print(distance*scaling_factor)
                #prob_dict = predict("every")
                prob_dict = predict(" ".join(subject))
                
                
                filtered_nouns = [x for x in nouns if x in prob_dict and x not in adjectives]
            #     nouns = [x for x in nouns if x not in adjectives]

                sorted_nouns = sorted(filtered_nouns,key=lambda x : prob_dict[x],reverse=True)
                print("SOME SORTED NOUNS:",sorted_nouns[:50])
                
                filtered_adjs = [x for x in adjectives if x in vecs and x in frequencies and x in prob_dict]
                freq_sorted_adjs = sorted(filtered_adjs,key = lambda x : frequencies[x],reverse=True)

                # quds = list(list(zip(*visualize_cosine(np.mean([vecs[subj],vecs[word]],axis=0),freq_sorted_adjs,vecs)))[0])
                quds = ['leafy','stable']
                # quds=quds[:quds.index('stable')]+['strong','stable','wooden','connected','leafy','unyielding','bearded']
                # quds = quds[:500:2]

                run = DistRSAInference(
                subject=subject,predicate=word,
                possible_utterances=sorted_nouns[:500]+[word],
        #         possible_utterances=sorted_nouns[:sorted_nouns.index(pred)+1],
            #     quds=['strong','stable','wooden','connected','leafy','unyielding','bearded'],
                quds = quds,
            #         ['unyielding']+list(list(zip(*visualize_cosine(np.mean([vecs['man'],vecs[word]],axis=0),freq_sorted_adjs,vecs)[:500:10]))[0]),

                # possible_utterances=nouns[:1000]+adjectives[:1000],
                object_name="animals_spec",
                mean_vecs=True,
                pca_remove_top_dims=False,
                sig1=5.0,sig2=tf.cast(scaling_factor*distance,dtype=tf.float32),
            #         proposed_sig2*scaling_factor,
                qud_weight=0.0,freq_weight=0.0,
                categorical="categorical",
                vec_length=300,vec_type="glove.6B.",
                sample_number = 500,
                number_of_qud_dimensions=1,
                burn_in=350,
                seed=False,trivial_qud_prior=False,
                step_size=0.01,
                frequencies=prob_dict,
                qud_prior_weight=0.5,
                rationality=0.99,
                speaker_world=vecs['the']
                )


                run.compute_results(load=0,save=False)
                print("QUD PROXIMAL")
                print(run.qud_results()[:20])
                # print(np.sum([x[1] for x in run.qud_results()]))
                # print("BASELINE MODEL")
                # print(run.world_movement("cosine",comparanda=[animal_feature for animal_feature in animal_features if animal_feature in vecs]))
                # print(run.world_movement("cosine",comparanda=[noun for noun in nouns if noun in vecs]))
                # print(run.world_movement("cosine",comparanda=[adjective for adjective in adjectives if adjective in vecs]))
                # print(run.world_movement("cosine",do_projection=True,comparanda=[animal_feature for animal_feature in animal_features if animal_feature in vecs]))
                # print(run.world_movement("euclidean",comparanda=[animal_feature for animal_feature in animal_features if animal_feature in vecs]))
                # # print(run.world_movement("euclidean",do_projection=True,comparanda=[animal_feature for animal_feature in animal_features if animal_feature in vecs]))
                # # print(run.world_movement("cosine",do_projection=True,comparanda=[noun for noun in nouns if noun in vecs]))
                # print(run.world_movement("euclidean",comparanda=[noun for noun in nouns if noun in vecs]))
                # # print(run.world_movement("euclidean",do_projection=True,comparanda=[noun for noun in nouns if noun in vecs]))
                # print(run.world_movement("cosine",comparanda=[adjective for adjective in adjectives if adjective in vecs]))
                # # print(run.world_movement("cosine",do_projection=True,comparanda=[adjective for adjective in adjectives if adjective in vecs]))
                # print(run.world_movement("euclidean",comparanda=[adjective for adjective in adjectives if adjective in vecs]))
                # print(run.world_movement("euclidean",do_projection=True,comparanda=[adjective for adjective in adjectives if adjective in vecs]))
                print(run.baseline_model('mean'))

# run()
# raise Exception


vecs = pickle.load(open("data/word_vectors/glove.twitter.27B.mean_vecs25",'rb'),encoding='latin1')
from utils.load_data import get_words
frequencies,nouns,adjectives,verbs = get_words(preprocess=False)
nouns,adjectives,verbs = sorted(nouns, key=lambda x: frequencies[x], reverse=True),sorted(adjectives, key=lambda x: frequencies[x], reverse=True),sorted(verbs, key=lambda x: frequencies[x], reverse=True)



for pred in ['sheep','lion','tree']:
    for subj in ["man"]:
        prob_dict = predict(" ".join(["that",subj, "is","a"]))
        filtered_nouns = [x for x in nouns if x in prob_dict and x not in adjectives and x not in verbs]
        sorted_nouns = sorted(filtered_nouns,key=lambda x : prob_dict[x],reverse=True)
        filtered_adjs = [x for x in adjectives if x in prob_dict and x not in nouns and x not in verbs]
        sorted_adjs = sorted(filtered_adjs,key=lambda x : prob_dict[x],reverse=True)

        adj_prob_dict = predict(" ".join([pred, "is", "a"]))

        # for sig1 in [5e-3,5e-4]:
        #     for sig2 in [5e-2,5e-3]:
        qud_frequencies = {}
        for word in [x for x in nouns+adjectives if x in vecs]:
            qud_frequencies[word] = scipy.spatial.distance.cosine(np.mean([vecs[pred],vecs[subj]],axis=0),vecs[word])
        #         for lr = [5e-5,5e-6]
        for i in range(2):
    #         ["the", subj, "is", "a"]

#             distance  = scipy.spatial.distance.cosine(vecs[subj],vecs[pred])
#             print(distance*scaling_factor)
            #prob_dict = predict("every")



    #         quds = list(list(zip(*visualize_cosine(np.mean([vecs[subj],vecs[pred]],axis=0),freq_sorted_adjs,vecs)))[0])
    #         quds = quds[:2*sorted_nouns.index(pred):2]
#             quds = ["evil","leggy","cool","plotting","small"] + list(list(zip(*visualize_cosine(np.mean([vecs[subj],vecs[pred]],axis=0),freq_sorted_adjs,vecs)))[0])[:10]
    #         quds=quds[:quds.index('stable')]+['strong','stable','wooden','connected','leafy','unyielding','bearded']

            quds = list(list(zip(*visualize_cosine(np.mean([vecs[pred],vecs[subj]],axis=0),[x for x in adjectives if x in vecs],vecs)))[0])
            quds = [qud for qud in quds if qud in adj_prob_dict and qud in vecs]
            quds = sorted(quds,key=lambda x : adj_prob_dict[x],reverse=True)
            quds = quds[:1000]
            # +['chestnut']
            # print("BASELINE",quds[:20])

            run = DistRSAInference(
            subject=[subj],predicate=pred,
            # possible_utterances=animals,
            # quds=animal_features,
            quds=quds,
            # quds = animal_features,
        #         ['unyielding']+list(list(zip(*visualize_cosine(np.mean([vecs['man'],vecs[word]],axis=0),freq_sorted_adjs,vecs)[:500:10]))[0]),

            possible_utterances=nouns[:1000]+[pred],
            # possible_utterances=
            # [noun for noun in nouns if noun not in adjectives][:100]+[adj for adj in adjectives if adj not in nouns][:100]+[pred],
            # sorted_nouns[sorted_nouns.index(pred) if pred in sorted_nouns else 500]+['horse'],
            object_name="animals_spec",
            mean_vecs=True,
            pca_remove_top_dims=True,
            sig1=0.005,sig2=0.05,
        #         proposed_sig2*scaling_factor,
            qud_weight=0.0,freq_weight=1.0,
            categorical="non-categorical",
            vec_length=25,vec_type="glove.twitter.27B.",
            sample_number = 5000,
            number_of_qud_dimensions=2,
            burn_in=4500,
            seed=False,trivial_qud_prior=False,
            step_size=0.0005,
            frequencies=prob_dict,
            qud_frequencies=qud_frequencies,
            qud_prior_weight=0.5,
            rationality=1.0,
            run_s2=False,
            speaker_world=vecs[subj]+(1/10*vecs["the"]),
            s1_only=False
            )
            run.compute_results(load=0,save=False)
            print(run.qud_results()[:20])
            print(run.world_movement("cosine",comparanda=[animal_feature for animal_feature in animal_features if animal_feature in vecs]))
            print(run.world_movement("cosine",do_projection=True,comparanda=[animal_feature for animal_feature in animal_features if animal_feature in vecs]))
            print(run.world_movement("euclidean",comparanda=[animal_feature for animal_feature in animal_features if animal_feature in vecs]))
            print(run.world_movement("euclidean",do_projection=True,comparanda=[animal_feature for animal_feature in animal_features if animal_feature in vecs]))

            print("BASELINE")
            print(run.baseline_model('mean')[:20])

            # raise Exception


