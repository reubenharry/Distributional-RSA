from dist_rsa.utils.helperfunctions import memoized_s2
from dist_rsa.utils.build_s2_dict import build_s2_dict
import pickle


build_s2_dict('man',['ox','nightmare','bag'])
raise Exception
rerun = True
if rerun:
    build_s2_dict('man',['ox','nightmare','bag'])

word_to_l1 = pickle.load(open("dist_rsa/data/results/pickles/word_to_l1",'rb'))


quds = set([item for subitem in list(zip(*word_to_l1[list(word_to_l1)[0]]))[0] for item in subitem])
print(quds)
# q1 = input('q1')
# q2 = input('q2')

q1 = 'evil'
q2 = 'mysterious'

q = [q1,q2]
q = sorted(q)

out = memoized_s2(q,word_to_l1)

print(out)


# from collections import defaultdict
# import pickle
# import scipy.stats
# from dist_rsa.utils.load_data import animals,animal_features
# from dist_rsa.utils.helperfunctions import memoized_s2
# # from dist_rsa.models.l1_cat_1d import l1_cat_1d
# # from dist_rsa.models.l1_cat_2d import l1_cat_2d
# # from dist_rsa.models.l1_cat_3d import l1_cat_3d
# # from dist_rsa.models.l1_cat_2d_v300 import l1_cat_2d_v300
# from dist_rsa.models.l1_cat_2d_memo import l1_cat_2d_memo
# # from dist_rsa.models.s2_cat_2d import s2_cat_2d
# # from dist_rsa.models.s2_cat_1d import s2_cat_1d
# from dist_rsa.utils.load_data import get_words
# import numpy as np

# nouns,adjs = get_words()
# concrete_threshold = 3.0
# abstract_threshold = 2.5

# vecs = pickle.load(open("dist_rsa/data/word_vectors/glove.6B.mean_vecs300",'rb'))
# # possible_utterances = sorted([n for n in nouns if nouns[n] > concrete_threshold and n in vecs],\
#                 # key=lambda x:scipy.spatial.distance.cosine(vecs[x],vecs['man']),reverse=False)[:50]
# possible_utterances = ["ox","nightmare","bag"]
# # quds = []
# # for utt in possible_utterances:
# #     quds += sorted([n for n in adjs if adjs[n] < abstract_threshold and n in vecs],\
# #                 key=lambda x:scipy.spatial.distance.cosine(vecs[x],vecs[utt]),reverse=False)[30:50]
# # quds = list(set(quds))
# # prob_dict = predict(" ".join(["men", "are",]))

# subj = 'man'

# quds = sorted([n for n in adjs if adjs[n] < abstract_threshold and n in vecs ],\
#     # key=lambda x:prob_dict[x],reverse=True)[:200]
#     key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[subj]],axis=0)),reverse=False)
# quds = quds[:50]


# print("S2 CHOSEN QUDS",quds)

# # l1_cat_2d(('man','shark'))
# # l1_cat_2d(('boat','knife'))
# # l1_cat_3d(('man','sheep'))
# # s2_cat_1d(possible_utterances)


# def make_utt_to_qud_dist(possible_utterances,model,name="word_to_l1"):

#     utt_to_qud_dist = {}
#     for poss_utt in possible_utterances:
#         utt_to_qud_dist[poss_utt] = sorted(model(("man",poss_utt),possible_utterances,quds),key=lambda x:x[0])
#         pickle.dump(utt_to_qud_dist,open("dist_rsa/data/results/pickles/"+name,'wb'))

# def find_rank_of_qud(q,l):
#     for x in l:
#         if x[0]==q:
#             return x[1] 
#     raise Exception(q,"not in list")

# def sort_words_by_qud(utt_to_qud_dist,q):
#     words = list(utt_to_qud_dist)
#     return sorted(words,key=lambda x: find_rank_of_qud(q,utt_to_qud_dist[x]),reverse=True)

# make_utt_to_qud_dist(["ox","nightmare","bag"],l1_cat_2d_memo,"test_run")


# utt_to_qud_dist = pickle.load(open("dist_rsa/data/results/pickles/test_run",'rb'))
# print(memoized_s2(["independent","dumb"],utt_to_qud_dist))
