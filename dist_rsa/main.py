from collections import defaultdict
import pickle
import scipy.stats
from dist_rsa.utils.load_data import animals,animal_features
from dist_rsa.utils.helperfunctions import memoized_s2
from dist_rsa.models.l1_cat_1d import l1_cat_1d
from dist_rsa.models.l1_cat_2d import l1_cat_2d
from dist_rsa.models.l1_cat_3d import l1_cat_3d
from dist_rsa.models.l1_cat_2d_v300 import l1_cat_2d_v300
from dist_rsa.models.l1_cat_2d_memo import l1_cat_2d_memo
from dist_rsa.models.s2_cat_2d import s2_cat_2d
from dist_rsa.models.s2_cat_1d import s2_cat_1d
from dist_rsa.models.test_run import test_run
from dist_rsa.utils.load_data import get_words
# possible_utterances = animals
nouns,adjs = get_words()
concrete_threshold = 3.0
vecs = pickle.load(open("dist_rsa/data/word_vectors/glove.6B.mean_vecs300",'rb'))
possible_utterances = sorted([n for n in nouns if nouns[n] > concrete_threshold and n in vecs],\
                key=lambda x:scipy.spatial.distance.cosine(vecs[x],vecs['man']),reverse=False)[:50]

# l1_cat_2d(('man','shark'))
# l1_cat_2d(('boat','knife'))
# l1_cat_3d(('man','sheep'))
# s2_cat_2d(["strong","stable"])
# s2_cat_1d(possible_utterances)


def make_utt_to_qud_dist(possible_utterances,model,name="word_to_l1"):

    utt_to_qud_dist = {}
    for poss_utt in possible_utterances:
        utt_to_qud_dist[poss_utt] = sorted(model(("man",poss_utt),possible_utterances),key=lambda x:x[0])
        pickle.dump(utt_to_qud_dist,open("dist_rsa/data/results/pickles/"+name,'wb'))

def find_rank_of_qud(q,l):
    for x in l:
        if x[0]==q:
            return x[1] 
    raise Exception(q,"not in list")

def sort_words_by_qud(utt_to_qud_dist,q):
    words = list(utt_to_qud_dist)
    return sorted(words,key=lambda x: find_rank_of_qud(q,utt_to_qud_dist[x]),reverse=True)

# make_utt_to_qud_dist(["oak","serpent","mud","grove","table","rock","weasel","rabbit","boat"],l1_cat_2d_memo,"test_run")

utt_to_qud_dist = pickle.load(open("dist_rsa/data/results/pickles/test_run",'rb'))
print(memoized_s2(["vast","unaware"],utt_to_qud_dist))


# out = sort_words_by_qud(utt_to_qud_dist,['powerful'])
# print(out)



# utts = [(x,scipy.stats.entropy(utt_to_qud_dist[x],utt_to_qud_dist['lion'])) for x in utt_to_qud_dist]
# # utts = sorted(utts,key=lambda x: scipy.stats.entropy(utt_to_qud_dist[x],utt_to_qud_dist['lion']))

# print(utts)
# out = scipy.stats.entropy(utt_to_qud_dist['lion'],utt_to_qud_dist['rose'])
# print(out)
# print("L1 OUTPUTS\n")
# for i in l1_outputs:
#     print(i)
#     print(l1_outputs[i])

# run_twod(metaphors)

#takes the qud output of l1 and gives dictionary with rank and density of each word
# def count_density_and_rank(l1_outputs):

#     out_dict = defaultdict(float)

#     for item in l1_outputs: 

#         l1_output = l1_outputs[item]

#         mets,densities = zip(*l1_output)

#         print("METS\n",mets)
#         # print("DENSITIES\n",densities)

#         words = [x for y in mets for x in y]
#         print("WORDS\n",words)
#         for word in words:
#             rank = mets.index([word])
#             density = densities[rank]
#             out_dict[word] += density

#     print(out_dict)

#     return sorted(list(out_dict),key=lambda x: out_dict[x],reverse=True)
# def floating(n):
#     floating_words = []

#     for item in l1_outputs:
#         for x in l1_outputs[item]:
#             floating_words.append(x[0][0])

#     new_floating_words = []

#     for x in floating_words:
#         #print(quds)
#         b = True
#         for y in l1_outputs:
#             quds,values = list(zip(*l1_outputs[y][:n]))
#             if [x] not in quds:
#                 b = False
#         if b == True:
#             new_floating_words.append(x)

            
#     return list(set(new_floating_words))

# floating_words = floating(150)

# words = []
# for x in l1_outputs[('man','lion')]:
#     if x[0][0] not in floating_words:
#         words.append(x)
# print(words)
# print("TOP FLOATING WORDS\n",out)

# print("man is a lion\n",l1_outputs[('man','lion')])

# a = l1_outputs[('man','lion')]
# for x in a:
#     if x[0][0] not in out[:10]:
#         print(x)

# for x in l1_outputs:
#     quds_and_densities = l1_outputs[x]
#     quds,densities = list(zip(*l1_outputs[x]))
#     print(x)
#     for i in out[:20]:
#         if [i] in quds:
#             del quds_and_densities[quds.index([i])]
#     print(quds_and_densities)




# metaphors = [('man','lion'),('love','poison'),('work','prison')]

# run metaphors
# record density and rank of each qud at each:
#     some algorithm for multidim quds: count every n-tuple it appears in
# sort to find floating words
# compare to l0 movement
# display dict of metaphor quds with floating words removed

