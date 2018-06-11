from __future__ import division
from collections import defaultdict
import scipy
import numpy as np
import pickle
import itertools
from dist_rsa.dbm import *
from dist_rsa.utils.load_data import *
from dist_rsa.utils.helperfunctions import *
from dist_rsa.utils.config import abstract_threshold,concrete_threshold


# vecs = load_vecs(mean=False,pca=False,vec_length=300,vec_type='word2vec')
# vec_size,vec_kind = 25,'glove.twitter.27B.'
vec_size,vec_kind = 300,'glove.6B.'
freqs = pickle.load(open('dist_rsa/data/google_freqs/freqs','rb'))
nouns,adjs = get_words(with_freqs=False)
vecs = load_vecs(mean=True,pca=False,vec_length=vec_size,vec_type=vec_kind) 

  
print("swims",np.linalg.norm(vecs['swims']))
print("swimmer",np.linalg.norm(vecs['swimmer']))
print("distance",scipy.spatial.distance.cosine(vecs['swims'],vecs['swimmer']))


# qud_words = [a for a in list(adjs) if a in vecs and a in vecs]
# quds = sorted(qud_words,key=lambda x:freqs[x],reverse=True)

# possible_utterance_nouns = [n for n in nouns if nouns[n] > concrete_threshold and n in vecs and n in vecs]
# possible_utterances = sorted(possible_utterance_nouns,key=lambda x:freqs[x],reverse=True)
# possible_utterances=possible_utterances[:100]

# quds = quds[:70]
# print(quds[:100])
# for vec in ['man','shark','swimmer']:
for vec in ['predator','swims']:
    vecs[vec] /= np.linalg.norm(vecs[vec])

print("VECTOR MEASUREMENTS")
print("man on swims",projection_into_subspace_np(np.expand_dims(vecs['man'],1),np.expand_dims(vecs['swims'],1)))
print("shark on swims",projection_into_subspace_np(np.expand_dims(vecs['shark'],1),np.expand_dims(vecs['swims'],1)))
print("swimmer on swims",projection_into_subspace_np(np.expand_dims(vecs['swimmer'],1),np.expand_dims(vecs['swims'],1)))
print("man on predator",projection_into_subspace_np(np.expand_dims(vecs['man'],1),np.expand_dims(vecs['predator'],1)))
print("shark on predator",projection_into_subspace_np(np.expand_dims(vecs['shark'],1),np.expand_dims(vecs['predator'],1)))
print("swimmer on predator",projection_into_subspace_np(np.expand_dims(vecs['swimmer'],1),np.expand_dims(vecs['predator'],1)))

def l1_model(subj,pred):

    print('abstract_threshold',abstract_threshold)
    print('concrete_threshold',concrete_threshold)

    # qud_words = [a for a in list(adjs) if adjs[a] < abstract_threshold and a in vecs]

    # quds = sorted(qud_words,\
    #     key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[pred]],axis=0)),reverse=False)
        # key=lambda x:freqs[x],reverse=True)

    # noun_words = [n for n in nouns if nouns[n] > concrete_threshold and n in vecs]
    # possible_utterance_nouns = sorted(noun_words,\
    #     # key=lambda x:freqs[x],reverse=True)
    #     key=lambda x: scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[subj]],axis=0)),reverse=False)
    # # possible_utterance_nouns = 
    # # break

    # # quds[:100]
    # possible_utterance_adjs = quds
    # possible_utterances = possible_utterance_nouns[start:stop]
    # +possible_utterance_adjs
    # quds = ["balloon","red"]
    quds = ["predator"]
    # possible_utterances = ["angry","frog"]
    # possible_utterances = ["wall","party"]

    possible_utterances = ["shark","swimmer"]

    for x in possible_utterances:
        if x not in vecs:
            # print(x,"not in vecs")
            possible_utterances.remove(x)
            # raise Exception("utterance not in vecs")

    print("QUDS",quds[:50]) 
    print("UTTERANCES:\n",possible_utterances[:20])


    params = Inference_Params(
        vecs=vecs,
        subject=[subj],predicate=pred,
        quds=quds,
        possible_utterances=list(set(possible_utterances).union(set([pred]))),
        sig1=1.0,sig2=1.0,l1_sig1=0.01,
        qud_weight=0.0,freq_weight=0.0,
        number_of_qud_dimensions=1,
        poss_utt_frequencies=defaultdict(lambda:1),
        qud_frequencies=defaultdict(lambda:1),
        rationality=1.0,
        norm_vectors=False,
        heatmap=False,
        resolution=Resolution(span=10,number=100),
        model_type="discrete_mixture",
        )

    run = Dist_RSA_Inference(params)
    run.compute_l1(load=0,save=False)


    return run.tf_results

    # world_means = run.world_samples
    # print(world_means[:5],"MEANS")

    # print(results[:5])

    # if not is_baseline:
    #     worldm = run.world_movement("cosine",comparanda=[x for x in qud_words if x in vecs])
    #     # print("\nworld\n",worldm[:5])
    # else: worldm = None
        # out.write("\nWORLD MOVEMENT:\n")
        # out.write(str(worldm))
    # print("WORLD MOVEMENT WITH PROJECTION\n:",run.world_movement("cosine",comparanda=[x for x in quds if x in vecs],do_projection=True)[:50])
    # print("BASELINE:\n",sorted(qud_words,\
    #     key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[pred]],axis=0)),reverse=False)[:5])

    # demarg = demarginalize_product_space(results)
    # print("\ndemarginalized:\n,",demarg[:5])
    # out.write("\ndemarginalized:\n")
    # out.write((str(demarg)))

    # params.number_of_qud_dimensions=1
    # run = Dist_RSA_Inference(params)
    # run.compute_l1(load=0,save=False)
    # results2 = run.qud_results()
    # # print("\n1d results\n",results2[:10])
    # one_d = results2
    # one_d=None

if __name__ == "__main__":

    # for x in range(1):
    #     l1_model(("father","shark",0.5,0.5,1.0,0,100,False))
    for x in range(1):
        # worlds,quds=l1_model(("wall","angry",1.0,1.0,1.0,0,1000,True))
        # print(quds[:10])
        # worlds,quds=l1_model(("wall","frog",1.0,1.0,1.0,0,1000,True))
        # print(quds[:10])

        # worlds,quds=l1_model(("angry","wall",1.0,1.0,1.0,0,1000,True))
        # print(quds[:10])
        # worlds,quds=l1_model(("frog","wall",1.0,1.0,1.0,0,1000,True))
        # print(quds[:10])

        means1,worlds,quds=l1_model(subj="man",pred="swimmer")
        print(quds[:10])
        # means2,worlds,quds=l1_model(subj="man",pred="shark")
        # print(quds[:10])

        # print(scipy.spatial.distance.cosine(vecs['man']-means1[0],vecs['man']-means2[0]))
        # worlds,quds=l1_model(("wall","frog",1.0,1.0,1.0,0,1000,True))
        # print(quds[:10])



