from __future__ import division
name = input("save to: ")

from collections import defaultdict
import scipy
import numpy as np
import pickle
import itertools
from dist_rsa.dbm import *
from dist_rsa.utils.load_data import *
from dist_rsa.utils.helperfunctions import *
from dist_rsa.lm_1b_eval import predict
from dist_rsa.utils.config import abstract_threshold,concrete_threshold
from dist_rsa.utils.distance_under_projection import distance_under_projection


vecs = load_vecs(mean=False,pca=False,vec_length=300,vec_type='word2vec')
nouns,adjs = get_words(with_freqs=False)

def l1_model(metaphor):
    vec_size,vec_kind = 200,'glove.twitter.27B.'
    subj,pred,sig1,sig2,is_baseline = metaphor

    print('abstract_threshold',abstract_threshold)
    print('concrete_threshold',concrete_threshold)

    qud_words = [a for a in list(adjs) if adjs[a] < abstract_threshold and a in vecs]
    quds = sorted(qud_words,\
        key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[pred]],axis=0)),reverse=False)
    possible_utterance_nouns = sorted([n for n in nouns if nouns[n] > concrete_threshold and n in vecs],\
        key=lambda x: scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[subj]],axis=0)),reverse=False)

    possible_utterance_adjs = quds
    quds = quds[:50]
    print("QUDS",quds[:50]) 
    possible_utterances = possible_utterance_adjs[:25]+possible_utterance_nouns[:50]

    # possible_utterances = ['ox','bag','nightmare']
    # possible_utterances = possible_utterance_adjs[:50]
    # +quds[:25]
    # possible_utterance_adjs[:50]+possible_utterance_nouns[:50]
    # possible_utterance_nouns[:4]

    real_vecs = load_vecs(mean=True,pca=True,vec_length=vec_size,vec_type=vec_kind)    

    for x in possible_utterances:
        if x not in real_vecs:
            print(x,"not in vecs")
            possible_utterances.remove(x)
            # raise Exception("utterance not in vecs")

    print("UTTERANCES:\n",possible_utterances[:20])

    params = Inference_Params(
        vecs=real_vecs,
        subject=[subj],predicate=pred,
        quds=quds,
        possible_utterances=list(set(possible_utterances).union(set([pred]))),
        sig1=sig1,sig2=sig2,
        qud_weight=0.0,freq_weight=0.0,
        categorical="categorical",
        sample_number = 1000,
        number_of_qud_dimensions=2,
        # burn_in=900,
        seed=False,trivial_qud_prior=False,
        step_size=1e-5,
        poss_utt_frequencies=defaultdict(lambda:1),
        qud_frequencies=defaultdict(lambda:1),
        qud_prior_weight=0.5,
        rationality=1.0,
        norm_vectors=False,
        variational=True,
        variational_steps=100,
        baseline=is_baseline
        # world_movement=True

        )

    run = Dist_RSA_Inference(params)

    run.compute_l1(load=0,save=False)

    results = run.qud_results()

    print(results[:5])

    if not is_baseline:
        worldm = run.world_movement("cosine",comparanda=[x for x in quds if x in real_vecs])
        print(worldm[:5])
        l1_dict["world"]=worldm
        # out.write("\nWORLD MOVEMENT:\n")
        # out.write(str(worldm))
    # print("WORLD MOVEMENT WITH PROJECTION\n:",run.world_movement("cosine",comparanda=[x for x in quds if x in real_vecs],do_projection=True)[:50])
    # print("BASELINE:\n",sorted(qud_words,\
    #     key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[pred]],axis=0)),reverse=False)[:5])

    print("RESULTS\n",[(x,np.exp(y)) for (x,y) in results[:5]])
    demarg = demarginalize_product_space(results)
    print("\ndemarginalized:\n,",demarg[:5])
    l1_dict['demarg']=demarg
    # out.write("\ndemarginalized:\n")
    # out.write((str(demarg)))
    return results

if __name__ == "__main__":

    l1_dict = {}
    qud_only_dict = {}

    # out = open("dist_rsa/data/l1_results_"+name,"w")
    # out.write("RESULTS 25D\n")
    for subj,pred in [("man","lion"),("woman","rose"),("place","junkyard")]:
    # test_metaphors:
    # 
        # out.write("\n"+subj+" is a "+pred)

        for sig1,sig2 in [(1.0,1.0)]:
            for is_baseline in [False,True]:
                # out.write('\n')
                # out.write("sig1/sig2 "+str(sig1)+"/"+str(sig2)+" baseline: "+str(is_baseline))
                if is_baseline:
                    # out.write('\n\nBASELINE')
                    results = l1_model((subj,pred,0.1,0.1,is_baseline))
                    # results = [(0.5,0.5)]
                    # out.write(str([(x,np.exp(y)) for (x,y) in results[:5]]))
                    qud_only_dict['baseline']=results
                    # print(baseline_dict)
                else:
                    # out.write("L1 RUN:\n")
                    for i in range(3):
                        results = l1_model((subj,pred,sig1,sig2,is_baseline))
                        # results = [(0.5,0.5)]
                        # out.write("\nL1:"+str(i))
                        # out.write('\n')
                        # out.write(str([(x,np.exp(y)) for (x,y) in results[:5]]))
                        l1_dict['l1'+str(i)]=results
                        # print(l1_dict)


    pickle.dump(l1_dict,open("l1_dict"+name,"wb"))
    pickle.dump(qud_only_dict,open("qud_only_dict"+name,"wb"))


