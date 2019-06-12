from collections import defaultdict
import pickle
import scipy.stats
from dist_rsa.utils.load_data import animals,animal_features
from dist_rsa.models.s2_cat_2d_memo import s2_cat_2d_memo
from dist_rsa.utils.load_data import get_words,load_vecs
from dist_rsa.utils.config import abstract_threshold,concrete_threshold
import numpy as np


def build_s2_dict(subj,possible_utterances):
	vecs = load_vecs(mean=True,pca=True,vec_length=300,vec_type='glove.6B.')
	nouns,adjs = get_words()

	# possible_utterances = sorted([n for n in nouns if nouns[n] > concrete_threshold and n in vecs],\
	#     key=lambda x: scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[subj]],axis=0)),reverse=False)
	qud_words = [a for a in list(adjs) if adjs[a] < abstract_threshold and a in vecs]
	quds = sorted(qud_words,\
    	key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[subj]],axis=0)),reverse=False)
	quds = quds[500:1000:10]

	# possible_utterance_nouns = sorted([n for n in nouns if nouns[n] > concrete_threshold and n in vecs],\
	# 	key=lambda x: scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[subj]],axis=0)),reverse=False)
	# possible_utterance_adjs = quds

	print("QUDS",quds) 
	# possible_utterances = possible_utterance_nouns[::100]+possible_utterance_adjs[::100]

	utt_to_qud_dist = {}
	for poss_utt in possible_utterances:
	    result = sorted(s2_cat_2d_memo((subj,poss_utt),quds,possible_utterances),key=lambda x:x[0])
	    # words,distribution = list(zip(*result))
	    # print(result[:10])
	    utt_to_qud_dist[poss_utt] = result


	pickle.dump(utt_to_qud_dist,open("dist_rsa/data/results/pickles/word_to_l1",'wb'))
	 
# utts = [(x,scipy.stats.entropy(utt_to_qud_dist[x],utt_to_qud_dist['lion'])) for x in utt_to_qud_dist]
# utts = sorted(utts,key=lambda x: scipy.stats.entropy(utt_to_qud_dist[x],utt_to_qud_dist['lion']))

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

