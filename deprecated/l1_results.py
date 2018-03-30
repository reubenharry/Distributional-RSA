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
vec_size,vec_kind = 200,'glove.twitter.27B.'
# freqs = pickle.load(open('dist_rsa/data/google_freqs/freqs','rb'))
nouns,adjs = get_words(with_freqs=False)
real_vecs = load_vecs(mean=True,pca=True,vec_length=vec_size,vec_type=vec_kind)  

# qud_words = [a for a in list(adjs) if a in vecs and a in real_vecs]
# quds = sorted(qud_words,key=lambda x:freqs[x],reverse=True)

# possible_utterance_nouns = [n for n in nouns if nouns[n] > concrete_threshold and n in vecs and n in real_vecs]
# possible_utterances = sorted(possible_utterance_nouns,key=lambda x:freqs[x],reverse=True)
# possible_utterances=possible_utterances[:100]

# quds = quds[:70]
# print(quds[:100])

def l1_model(metaphor):
    subj,pred,sig1,sig2,is_baseline = metaphor

    print('abstract_threshold',abstract_threshold)
    print('concrete_threshold',concrete_threshold)

    qud_words = [a for a in list(adjs) if adjs[a] < abstract_threshold and a in vecs]

    quds = sorted(qud_words,\
        key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[pred]],axis=0)),reverse=False)
        # key=lambda x:prob_dict[x],reverse=True)


    possible_utterance_nouns = sorted([n for n in nouns if nouns[n] > concrete_threshold and n in vecs],\
        # key=lambda x:prob_dict[x],reverse=True)
        key=lambda x: scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[subj]],axis=0)),reverse=False)
    # possible_utterance_nouns = 
    # break
    quds = quds[:50]
    possible_utterance_adjs = quds
    possible_utterances = possible_utterance_nouns[:50]+possible_utterance_adjs


    for x in possible_utterances:
        if x not in real_vecs:
            print(x,"not in vecs")
            possible_utterances.remove(x)
            # raise Exception("utterance not in vecs")

    print("QUDS",quds[:50]) 
    print("UTTERANCES:\n",possible_utterances[:20])


    params = Inference_Params(
        vecs=real_vecs,
        subject=[subj],predicate=pred,
        quds=quds,
        possible_utterances=list(set(possible_utterances).union(set([pred]))),
        sig1=sig1,sig2=sig2,l1_sig1=1.0,
        qud_weight=0.0,freq_weight=0.0,
        categorical="categorical",
        sample_number = 1000,
        number_of_qud_dimensions=2,
        # burn_in=900,
        seed=False,trivial_qud_prior=False,
        step_size=1e-6,
        poss_utt_frequencies=defaultdict(lambda:1),
        qud_frequencies=defaultdict(lambda:1),
        qud_prior_weight=0.5,
        rationality=1.0,
        norm_vectors=False,
        variational=True,
        variational_steps=300,
        baseline=is_baseline
        # world_movement=True
        )

    run = Dist_RSA_Inference(params)
    run.compute_l1(load=0,save=False)
    results = run.qud_results()

    print(results[:5])

    if not is_baseline:
        worldm = run.world_movement("cosine",comparanda=[x for x in qud_words if x in real_vecs])
        print("\nworld\n",worldm[:5])
    else: worldm = None
        # out.write("\nWORLD MOVEMENT:\n")
        # out.write(str(worldm))
    # print("WORLD MOVEMENT WITH PROJECTION\n:",run.world_movement("cosine",comparanda=[x for x in quds if x in real_vecs],do_projection=True)[:50])
    # print("BASELINE:\n",sorted(qud_words,\
    #     key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[pred]],axis=0)),reverse=False)[:5])

    print("RESULTS\n",[(x,np.exp(y)) for (x,y) in results[:5]])
    demarg = demarginalize_product_space(results)
    print("\ndemarginalized:\n,",demarg[:5])
    # out.write("\ndemarginalized:\n")
    # out.write((str(demarg)))

    params.number_of_qud_dimensions=1
    run = Dist_RSA_Inference(params)
    run.compute_l1(load=0,save=False)
    results2 = run.qud_results()
    print("\n1d results\n",results2[:10])
    one_d = results2
    # one_d=None



    return results,demarg,worldm,one_d

if __name__ == "__main__":

    results_dict = {}
    results_dict['l1_dict'] = {}
    results_dict['qud_only_dict'] = {}

    # out = open("dist_rsa/data/l1_results_"+name,"w")
    # out.write("RESULTS 25D\n")
    for subj,pred in test_metaphors:
    # [("man","lion")]:
    # ,("lion","man"),("woman","lion"),("woman","rose"),("love","poison")]:
    # ,("woman","lion"),("woman","rose"),("place","junkyard")]:
        results_dict['l1_dict'][(subj,pred)]={}
        results_dict['qud_only_dict'][(subj,pred)]={}
    # test_metaphors:
    # 
        # out.write("\n"+subj+" is a "+pred)

        for sig1,sig2 in [(0.1,0.1)]:
            for is_baseline in [False,True]:
                # out.write('\n')
                # out.write("sig1/sig2 "+str(sig1)+"/"+str(sig2)+" baseline: "+str(is_baseline))
                if is_baseline:
                    # out.write('\n\nBASELINE')
                    results,demarg,worldm,one_d = l1_model((subj,pred,0.1,0.1,is_baseline))
                    # results = [(0.5,0.5)]
                    # out.write(str([(x,np.exp(y)) for (x,y) in results[:5]]))
                    results_dict['qud_only_dict'][(subj,pred)]['baseline']=results
                    results_dict['qud_only_dict'][(subj,pred)]['1d']=one_d
                    # print(baseline_dict)
                else:
                    # out.write("L1 RUN:\n")
                    for i in range(3):
                        results,demarg,worldm,one_d = l1_model((subj,pred,sig1,sig2,is_baseline))
                        # results = [(0.5,0.5)]
                        # out.write("\nL1:"+str(i))
                        # out.write('\n')
                        # out.write(str([(x,np.exp(y)) for (x,y) in results[:5]]))
                        results_dict['l1_dict'][(subj,pred)]['l1'+str(i)]=results
                        results_dict['l1_dict'][(subj,pred)]['demarg'+str(i)]=demarg
                        results_dict['l1_dict'][(subj,pred)]["world"+str(i)]=worldm
                        results_dict['l1_dict'][(subj,pred)]["1d"+str(i)]=one_d


                        # print(l1_dict)


    pickle.dump(results_dict,open("results_dict_"+name,"wb"))


