from collections import defaultdict
import pickle
import numpy as np
import scipy.stats
import scipy
from dist_rsa.utils.load_data import animals,animal_features,controls,metaphors,sentences
from dist_rsa.models.l1_cat_1d import l1_cat_1d
from dist_rsa.models.l1_cat_2d import l1_cat_2d
from dist_rsa.models.l1_cat_2d_300_best import l1_cat_2d_300_best
from dist_rsa.models.l1_cat_2d_300_best_2 import l1_cat_2d_300_best_2
from dist_rsa.models.l1_cat_2d_memo import l1_cat_2d_memo
from dist_rsa.utils.load_data import get_words
from utils.load_AN_phrase_data import load_AN_phrase_data
from dist_rsa.utils.helperfunctions import metaphors_to_csv,metaphors_to_html

# possible_utterances = animals
name='short_2'

nouns,adjs = get_words()
concrete_threshold = 3.0
abstract_threshold = 2.5

vecs = pickle.load(open("dist_rsa/data/word_vectors/glove.6B.mean_vecs300",'rb'))
possible_utterances = sorted([n for n in nouns if nouns[n] > concrete_threshold and n in vecs],\
                key=lambda x:scipy.spatial.distance.cosine(vecs[x],vecs['man']),reverse=False)[:50]



print(metaphors)
# raise Exception
# for utt in ["lion","shark","sheep",'nightmare','angel']:
#     results = l1_cat_2d(('man',utt))
#     print("RESUTLS\n\n\n",[(x,np.exp(y)) for (x,y) in results][:50])

# metaphors =[('oven','fiery'),('mood','fiery')]

# metaphors = load_AN_phrase_data(real=True)[:20]

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

make = True

if make:

    l1 = make_l1_dict(metaphors)
    baseline = make_baseline_dict(metaphors)
    pickle.dump(baseline,open("dist_rsa/data/baseline_dict"+name,'wb'))
    pickle.dump(l1,open("dist_rsa/data/l1_dict"+name,'wb'))


l1 = pickle.load(open("dist_rsa/data/l1_dict"+name,'rb'))
baseline = pickle.load(open("dist_rsa/data/baseline_dict"+name,'rb'))

f = open("dist_rsa/data/metaphor_file",'w')
for metaphor in metaphors:
    metaphor = tuple(metaphor)
    print('\n\n\n',metaphor)
    print(baseline[metaphor])
    print(l1[metaphor])
    f.write('\n\n\n'+str(metaphor))
    f.write('\nbaseline:\n'+str(baseline[metaphor]))
    f.write('\l1 predictions:\n'+str(l1[metaphor]))
f.close()



select_metaphors = zip(sentences,metaphors[:15])
# metaphors_to_html(select_metaphors,l1,baseline,controls)


# "The man is a lion.","Love is a poison."]

# metaphors_to_csv(select_metaphors,l1,baseline,controls)
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

