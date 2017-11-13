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


vecs = load_vecs(mean=True,pca=True,vec_length=300,vec_type='glove.6B.')
nouns,adjs = get_words()

def l1_model(metaphor):
    vec_size,vec_kind = 25,'glove.twitter.27B.'
    subj,pred,sig1,sig2,l1_sig1,start,stop,is_baseline,qud_num = metaphor

    print('abstract_threshold',abstract_threshold)
    print('concrete_threshold',concrete_threshold)

    quds = ['strong','fast','green']
    possible_utterances = ['oak','horse']
    # possible_utterances = ['ox','bag','nightmare']
    # possible_utterances = possible_utterance_adjs[:50]
    # +quds[:25]
    # possible_utterance_adjs[:50]+possible_utterance_nouns[:50]
    # possible_utterance_nouns[:4]


    real_vecs = load_vecs(mean=True,pca=False,vec_length=vec_size,vec_type=vec_kind)    
    print("unyielding in real vecs","unyielding" in real_vecs)

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
        sig1=sig1,sig2=sig2, l1_sig1=l1_sig1,
        qud_weight=0.0,freq_weight=0.0,
        categorical="categorical",
        sample_number = 1000,
        number_of_qud_dimensions=1,
        # burn_in=900,
        seed=False,trivial_qud_prior=False,
        step_size=1e-4,
        poss_utt_frequencies=defaultdict(lambda:1),
        qud_frequencies=defaultdict(lambda:1),
        qud_prior_weight=0.5,
        rationality=0.99,
        norm_vectors=False,
        variational=True,
        variational_steps=1000,
        baseline=is_baseline
        # world_movement=True

        )

    run = Dist_RSA_Inference(params)

    run.compute_l1(load=0,save=False)

    results = run.qud_results()
    if not is_baseline:
        # worldm = run.world_movement("cosine",comparanda=[x for x in quds if x in real_vecs])    
        worldm = run.world_movement("cosine",comparanda=[x for x in quds if x in real_vecs],do_projection=False)
    else: worldm=None
    # print("WORLD MOVEMENT WITH PROJECTION\n:",run.world_movement("cosine",comparanda=[x for x in quds if x in real_vecs],do_projection=True)[:50])
    # print("BASELINE:\n",sorted(quds,\
    #     key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[pred]],axis=0)),reverse=False)[:5])
    if not is_baseline:
        new_results = []
        for x,y in results:
            
            # print(x)
            subj_proj = projection_debug(run.inference_params.subject_vector,np.expand_dims(real_vecs[x[0]],-1))
            result_proj = np.expand_dims(projection_debug(np.mean(run.world_samples,axis=0),np.expand_dims(real_vecs[x[0]],-1)),0)
        # print("\ndemarginalized:\n",demarginalize_product_space(results)[:5])
            new_results.append((x,np.exp(y),result_proj-subj_proj))
    else: new_results = [(x,np.exp(y)) for (x,y) in results]

    return new_results,worldm

if __name__ == "__main__":

    l1_model(("man","shark"))
    l1_model(("man","shark"))
    # l1_model(("tree","man",True))
    # l1_model(("tree","woman",False))

    # l1_model(("man","oak",True))
    # l1_model(("oak","tree",True))
    # l1_model(("tree","oak",True))
    # l1_model(("drug","athletics",False))
    # l1_model(("junkyard","place",True))
    # l1_model(("junkyard","place",False))

