from collections import defaultdict
import pickle
import numpy as np
import scipy.stats
import scipy
from dist_rsa.utils.load_data import animals,animal_features
from dist_rsa.models.l1_cat_1d import l1_cat_1d
from dist_rsa.models.l1_cat_2d import l1_cat_2d
from dist_rsa.models.l1_cat_baseline import l1_cat_baseline

# from dist_rsa.models.l1_cat_3d import l1_cat_3d
# from dist_rsa.models.l1_cat_2d_v300 import l1_cat_2d_v300
# from dist_rsa.models.l1_cat_2d_memo import l1_cat_2d_memo
from dist_rsa.utils.load_data import get_words
from utils.load_AN_phrase_data import load_AN_phrase_data
from dist_rsa.utils.helperfunctions import metaphors_to_csv

# possible_utterances = animals
nouns,adjs = get_words()
concrete_threshold = 3.0
abstract_threshold = 2.5

vecs = pickle.load(open("dist_rsa/data/word_vectors/glove.6B.mean_vecs300",'rb'))
possible_utterances = sorted([n for n in nouns if nouns[n] > concrete_threshold and n in vecs],\
                key=lambda x:scipy.spatial.distance.cosine(vecs[x],vecs['man']),reverse=False)[:50]

twitter_vecs = pickle.load(open("dist_rsa/data/word_vectors/glove.twitter.27B.mean_vecs25",'rb'))

metaphors = [('brain','muscle'),('brain','computer'),('brain','giant'),('life','dream'),
    ('life','gift'),('life','party'),('book','hug'),('body','city'),('biking','gateway'),
    ('bed','heaven'),('athletics','drug'),('ascent','journey'),
    ('room','dungeon'),('room','furnace'),('room','matchbox'),('room','wok'),
    ('rose','woman'),('school','game'),('ship','palace'),('sin','sickness'),('skin','parchment'),
    ('market','graveyard'),('sun','hammer'),('photographer','fly'),('place','junkyard'),('pornography','cancer'),
    ('principal','dictator'),('pub','family'),('life','train'),('life','snowflake'),('life','river'),('mother','mosquito'),
    ('music','language'),('relationship','treasure'),('voice','river'),('tongue','razor')]

metaphors = [x for x in metaphors if x[0] in twitter_vecs and x[1] in twitter_vecs]
metaphors = sorted(metaphors,key=lambda x:scipy.spatial.distance.cosine(twitter_vecs[x[0]],twitter_vecs[x[1]]) ,reverse=True)

print(metaphors)
# raise Exception
# for utt in ["lion","shark","sheep",'nightmare','angel']:
#     results = l1_cat_2d(('man',utt))
#     print("RESUTLS\n\n\n",[(x,np.exp(y)) for (x,y) in results][:50])

# metaphors =[('oven','fiery'),('mood','fiery')]

# metaphors = load_AN_phrase_data(real=True)[:20]

def make_simple_l1_dict(metaphors):
    l1_dict = {}
    for metaphor in metaphors:
        metaphor = tuple(metaphor)
        results = l1_cat_baseline(metaphor)
        results = list(zip(*results))[0]
        results = [x for y in results for x in y]
        print(results[:3])
        l1_dict[metaphor]=results[:10]
    return l1_dict

def make_l1_dict(metaphors):
    l1_dict = {}
    for metaphor in metaphors:
        metaphor = tuple(metaphor)
        results = l1_cat_2d(metaphor)
        results = list(zip(*results))[0]
        results = [x for y in results for x in y]
        print(results[:3])
        l1_dict[metaphor]=results[:10]
    return l1_dict

def make_baseline_dict(metaphors):
    l1_dict = {}
    for metaphor in metaphors:
        metaphor = tuple(metaphor)
        subj,pred = metaphor
        qud_words = [a for a in list(adjs) if adjs[a] < abstract_threshold and a in vecs]
        results = sorted(qud_words,\
            key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[pred]],axis=0)),reverse=False)
        # print(results[:3])
        l1_dict[metaphor]=results[:10]
    return l1_dict

# l1 = make_l1_dict(metaphors)
# baseline = make_baseline_dict(metaphors)
# pickle.dump(baseline,open("dist_rsa/data/baseline_dict",'wb'))
# pickle.dump(l1,open("dist_rsa/data/l1_dict",'wb'))


l1 = pickle.load(open("dist_rsa/data/l1_dict",'rb'))
baseline = pickle.load(open("dist_rsa/data/baseline_dict",'rb'))

# for metaphor in metaphors:
#     metaphor = tuple(metaphor)
#     print('\n\n\n',metaphor)
#     print(baseline[metaphor])
#     print(l1[metaphor])


select_metaphors = metaphors[:15]


print(metaphors_to_csv(select_metaphors,l1,baseline))

# results = l1_cat_3d(('man','sheep'))




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

