from dist_rsa.dbm import *
from dist_rsa.utils.load_data import *
from dist_rsa.models.l1 import l1_model

mean_center = True
remove_top_dims = False
norm_vectors = False
sig1 = 1.0
sig2 = 0.1
l1_sig1 = 0.1
hyperparams = Hyperparams(mean_center=mean_center,remove_top_dims=remove_top_dims,norm_vectors=norm_vectors,sig1=sig1,sig2=sig2,l1_sig1=l1_sig1)

r = Results_Pickler(path="dist_rsa/data/results/pickles/s2memo/"+hyperparams.show())
r.open()
print("\nHYPERPARAMS",hyperparams.show())

def utility(subj,pred,qud):

    results = dict(sorted(list(zip(r.results_dict[(subj,pred)].quds, r.results_dict[(subj,pred)].qud_marginals)),key=lambda x : x[1],reverse=True))
    print("pred:",pred, "results:",results)
    return results[qud]


words = ["shark","banana"]
scores = []
for word in words:
    scores.append(utility(subj="man",pred=word,qud="vicious"))

scores = scores / np.sum(scores)

print(sorted(list(zip(words,scores)),key = lambda x : x[0],reverse=True))

# print(utility(subj="man",pred="shark",qud="vicious"))

# for metaphor in r.results_dict:

#     # print(sorted(list(zip(r.results_dict[metaphor].quds, r.results_dict[metaphor].qud_marginals)),key=lambda x : x[1],reverse=True)[:5])
#     # results = dict(sorted(list(zip(r.results_dict[metaphor].quds, r.results_dict[metaphor].qud_marginals)),key=lambda x : x[1],reverse=True))
#     results = sorted(list(zip(r.results_dict[metaphor].quds, r.results_dict[metaphor].qud_marginals)),key=lambda x : x[1],reverse=True)
#     results = [result[0] for result in results]

#     results = results[:len(control_set[metaphor])]
#     out = len(set(results).intersection(set(control_set[metaphor])))
#     score += out
#     print("METAPHOR:",metaphor)
#     print(sorted(list(zip(r.results_dict[metaphor].quds, r.results_dict[metaphor].qud_marginals)),key=lambda x : x[-1],reverse=True)[:5])
#     # print(out)
#     # r.results_dict[metaphor].marginal_means-(r.results_dict[metaphor].subspace_prior_means[:,0])

# print("Score", score,"\n")

# from __future__ import division
# import scipy
# import numpy as np
# import pickle
# from dbm import *
# from collections import defaultdict
# from dist_rsa.utils.load_data import *
# from dist_rsa.utils.helperfunctions import *
# from dist_rsa.utils.load_data import get_freqs


# vecs = pickle.load(open("dist_rsa/data/word_vectors/glove.6B.mean_vecs300",'rb'),encoding='latin1')
# # from utils.load_data import get_words
# # frequencies,nouns,adjectives,verbs = get_words(preprocess=False)
# # adjectives = list(set(open("data/adjectives.txt","r").read().split("\n")))
# # nouns,adjectives,verbs = sorted(nouns, key=lambda x: frequencies[x], reverse=True),sorted(adjectives, key=lambda x: frequencies[x], reverse=True),sorted(verbs, key=lambda x: frequencies[x], reverse=True)
# nouns,adjs = get_words()

# def s2_cat_1d(possible_utterances):

#     for s2_qud in [['evil'],['cunning'],['stable'],['huge'],['fertile'],['boring']]:
#         for vec_size,vec_kind in [(300,'glove.6B.')]:
#     # ,(200,'glove.twitter.27B.')]:
#     # ,(50,'glove.6B.'),(300,'glove.6B.'),(300,'glove.840B.')]:
#             #         ["the", subj, "is", "a"]
#             subj = 'man'

#             abstract_threshold = 2.5
#             print('abstract_threshold',abstract_threshold)
#             concrete_threshold = 4.0
#             print('concrete_threshold',concrete_threshold)

#             qud_words = [a for a in list(adjs) if adjs[a] < abstract_threshold and a in vecs]
#             quds = sorted(qud_words,\
#                 key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[subj]],axis=0)),reverse=False)
#             # possible_utterance_nouns = sorted([n for n in nouns if nouns[n] > concrete_threshold and n in vecs],\
#             #     key=lambda x:scipy.spatial.distance.cosine(vecs[x],vecs[subj]),reverse=False)
#             # possible_utterance_adjs = sorted([n for n in adjs if adjs[n] > concrete_threshold and n in vecs],\
#             #     key=lambda x:scipy.spatial.distance.cosine(vecs[x],vecs[subj]),reverse=False)

#             quds=quds[:1000]

#             print("QUDS:\n",quds[:200])
#             print("UTTERANCES:\n",possible_utterances[:200])
            
#             real_vecs = pickle.load(open("dist_rsa/data/word_vectors/"+vec_kind+"pca and mean"+str(vec_size),'rb'),encoding='latin1')

#             run = DistRSAInference(
#             subject=[subj],predicate=possible_utterances[0],
#             # possible_utterances=animals,
#             # quds=animal_features,
#             quds=quds,
#             # quds = animal_features,
#         #         ['unyielding']+list(list(zip(*visualize_cosine(np.mean([vecs['man'],vecs[word]],axis=0),freq_sorted_adjs,vecs)[:500:10]))[0]),

#             possible_utterances=possible_utterances,
#             # possible_utterances=
#             # [noun for noun in nouns if noun not in adjectives][:100]+[adj for adj in adjectives if adj not in nouns][:100]+[pred],
#             # sorted_nouns[sorted_nouns.index(pred) if pred in sorted_nouns else 500]+['horse'],
#             object_name="animals_spec",
#             mean_vecs=True,
#             pca_remove_top_dims=True,
#             sig1=0.0005,sig2=0.005,
#         #         proposed_sig2*scaling_factor,
#             qud_weight=0.0,freq_weight=0.0,
#             categorical="categorical",
#             vec_length=vec_size,vec_type=vec_kind,
#             sample_number = 100,
#             number_of_qud_dimensions=1,
#             burn_in=80,
#             seed=False,trivial_qud_prior=False,
#             step_size=0.0005,
#             frequencies=defaultdict(lambda:1),
#             qud_frequencies=defaultdict(lambda:1),
#             qud_prior_weight=0.5,
#             rationality=1.0,
#             run_s2=True,
#             speaker_world=s2_qud,
#             s1_only=False,
#             norm_vectors=False
#             )
#             # print(real_vecs[subj],real_vecs[pred])

#             run.compute_results(load=0,save=False)
#             # print(run.world_movement("cosine",do_projection=True,comparanda=[x for x in abstract_adjs+abstract_nouns if x in real_vecs]))
#             # print("QUDS:\n",run.qud_results()[:20])
#             # print("WORLD MOVEMENT\n:",run.world_movement("cosine",comparanda=[x for x in quds if x in real_vecs])[:50])
#             # print("WORLD MOVEMENT\n:",run.world_movement("euclidean",comparanda=[x for x in quds if x in real_vecs])[:50])
#             # print("BASELINE:\n:",run.baseline_model('mean')[:20])

#             # raise Exception