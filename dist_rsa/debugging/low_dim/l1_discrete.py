from collections import defaultdict
import scipy
import numpy as np
import pickle
import itertools
from dist_rsa.dbm import *
from dist_rsa.utils.helperfunctions import *
from dist_rsa.utils.config import abstract_threshold,concrete_threshold
import edward as ed
from dist_rsa.utils.simple_vecs import real_vecs as simple_vecs
import itertools




def l1_model(subj,pred,sig1,sig2,l1_sig1,mixture_variational,resolution,quds,only_trivial,just_s1,just_l0,possible_utterances,discrete,variational,step_size):
    vec_size,vec_kind = 25,'glove.twitter.27B.'

    # print('abstract_threshold',abstract_threshold)
    # print('concrete_threshold',concrete_threshold)

    # quds = ['old','strong','stupid',"fast","green"]
    # possible_utterances = ["wall","tree","man","idiot"]
    # possible_utterances = ['ox','bag','nightmare']
    # possible_utterances = possible_utterance_adjs[:50]
    # +quds[:25]
    # possible_utterance_adjs[:50]+possible_utterance_nouns[:50]
    # possible_utterance_nouns[:4]

    vecs = simple_vecs
    real_vecs= simple_vecs

    # real_vecs = load_vecs(mean=True,pca=False,vec_length=vec_size,vec_type=vec_kind)
    # real_vecs['subj1']=real_vecs["pebble"]
    # real_vecs['subj2']=real_vecs["many"]
    # real_vecs['pred1']=real_vecs["myth"]

    # vecs['subj1']=vecs["many"]
    # vecs['subj2']=vecs["pebble"]
    # vecs['pred1']=vecs["myth"]


    # quds = list(adjs)[:20]
    # possible_utterances = list(nouns)[:200]  
        

    # possible_utterances = ["pred1","pred2"]


    # print("unyielding in real vecs","unyielding" in real_vecs)

    for x in possible_utterances:
        if x not in real_vecs:
            print(x,"not in vecs")
            possible_utterances.remove(x)
            # raise Exception("utterance not in vecs")

    print("UTTERANCES:\n",sorted(list(set(possible_utterances).union(set([pred]))))[:20])


    params = Inference_Params(
        vecs=real_vecs,
        subject=[subj],predicate=pred,
        quds=quds,
        possible_utterances=sorted(list(set(possible_utterances).union(set([pred])))),
        sig1=sig1,sig2=sig2, l1_sig1=l1_sig1,
        qud_weight=0.0,freq_weight=0.0,
        categorical="categorical",
        sample_number = 50,
        number_of_qud_dimensions=1,
        burn_in=1000,
        seed=False,trivial_qud_prior=False,
        step_size=step_size,
        poss_utt_frequencies=defaultdict(lambda:1),
        qud_frequencies=defaultdict(lambda:1),
        qud_prior_weight=0.5,
        rationality=1.0,
        norm_vectors=False,
        variational=variational,
        variational_steps=1000,
        baseline=False,
        discrete_l1=discrete,
        resolution=resolution,
        only_trivial=only_trivial,
        just_s1=just_s1,
        just_l0=just_l0,
        target_qud=0,
        mixture_variational=mixture_variational,

        # world_movement=True

        )

    run = Dist_RSA_Inference(params)

    run.compute_l1(load=0,save=False)


    # world_samples = tf_results[0]

    # print(world_samples,world_samples.shape)

    # print(tf_results)

    # pickle.dump(world_samples,open("dist_rsa/data/heatmap_samples.pkl",'wb'))
    # if discrete:
    tf_results = run.tf_results
    return tf_results
    # else: return run.world_samples,run.qud_samples

if __name__ == "__main__":

    quds=['vicious','swims']

    # a = l1_model(subj="man",pred="shark",sig1=1.0,sig2=1.0,l1_sig1=1.0,resolution=(100,0.01),quds=quds,possible_utterances=["shark","swimmer"], only_trivial=False,just_s1=False,just_l0=False,discrete=True,variational=False,step_size=1e-10,mixture_variational=False)
    # print([(x,np.exp(y)) for (x,y) in a[1]])

    a = l1_model(subj="man",pred="shark",sig1=1.0,sig2=0.1,l1_sig1=1.0,resolution=(100,0.01),quds=['vicious','swims'],possible_utterances=["shark","swimmer"], only_trivial=False,just_s1=False,just_l0=False,discrete=True,variational=False,step_size=1e-10,mixture_variational=False)
    print(a[1]) 


    # a = l1_model(subj="man",pred="swimmer",sig1=1.0,sig2=0.1,l1_sig1=1.0,resolution=(100,0.01),quds=["vicious"],possible_utterances=["shark","swimmer","man"], only_trivial=False,just_s1=False,just_l0=False,discrete=True,variational=True,step_size=1e-10,mixture_variational=False)
    # print([(x,np.exp(y)) for (x,y) in a[1]]) 
    # print([(x,np.exp(y)) for (x,y) in a[1]]) 

    # a = l1_model(subj="man",pred="swimmer",sig1=1.0,sig2=1.0,l1_sig1=1.0,resolution=(100,0.01),quds=quds,possible_utterances=["shark","swimmer"], only_trivial=False,just_s1=False,just_l0=False,discrete=False,variational=True,step_size=1e-10,mixture_variational=True)
    # print([(x,np.exp(y)) for (x,y) in a[1]]) 
    # print(a[1])
    # print([(quds[i],np.exp(x)) for (i,x) in enumerate(a[1])])
    # a = l1_model(subj="man",pred="shark",sig1=1.0,sig2=1.0,l1_sig1=1.0,resolution=(100,0.01),quds=quds,possible_utterances=["man","shark","swimmer"], only_trivial=False,just_s1=False,just_l0=False,discrete=False,variational=True,step_size=1e-10,mixture_variational=True)
    # print(a[1])
    # print([(quds[i],np.exp(x)) for (i,x) in enumerate(a[1])])
    # results_dict={}
    # # for sig1,sig2,l1_sig1 in itertools.product([1.0,10.0,0.1],[1.0,10.0,0.1],[1.0,10.0,0.1]):


    # for sig1,sig2,l1_sig1 in [[0.1,1.0,1.0],[1.0,1.0,1.0],[10.0,1.0,1.0],[100.0,1.0,1.0],
    #     [1.0,0.1,1.0],[1.0,1.0,1.0],[1.0,10.0,1.0],[1.0,100.0,1.0],
    #     [1.0,1.0,0.1],[1.0,1.0,1.0],[1.0,1.0,10.0],[1.0,1.0,100.0]]:

    # # for sig1,sig2,l1_sig1 in itertools.product([1.0],[1.0],[1.0]):

    #     a = l1_model(subj="man",pred="shark",sig1=sig1,sig2=sig2,l1_sig1=l1_sig1,resolution=(100,0.01),quds=quds,possible_utterances=["shark","swimmer"], only_trivial=False,just_s1=False,just_l0=False,discrete=True,variational=False)
    #     results_dict[(sig1,sig2,l1_sig1,"disc",0)]=list(zip(quds,np.exp(a[1])))
    #     for i in range(5):
    #         b = l1_model(subj="man",pred="shark",sig1=sig1,sig2=sig2,l1_sig1=l1_sig1,resolution=(100,0.01),quds=quds,possible_utterances=["shark","swimmer"], only_trivial=False,just_s1=False,just_l0=False,discrete=False,variational=True)
    #         c = l1_model(subj="man",pred="shark",sig1=sig1,sig2=sig2,l1_sig1=l1_sig1,resolution=(100,0.01),quds=quds,possible_utterances=["shark","swimmer"], only_trivial=False,just_s1=False,just_l0=False,discrete=False,variational=False)
    #         results_dict[(sig1,sig2,l1_sig1,"var",i)]=b[1]
    #         results_dict[(sig1,sig2,l1_sig1,"hmc",i)]=c[1]

    # pickle.dump(results_dict,open("dist_rsa/data/discrete_sig_search",'wb'))

    # print(a[1])
        # print("sig1",sig1,"sig2",sig2,"l1_sig1",l1_sig1, list(zip(quds,np.exp(a[1]))))
    # print([(x[0],np.exp(x[1])) for x in a[1]])
    # b = l1_model(s ubj="subj1",pred="pred2",sig1=0.1,sig2=0.1,l1_sig1=10.0,resolution=(100,0.1),quds=['qud1','qud2'],only_trivial=False,just_s1=False,just_l0=False)
    # print(b[-1])
    # l1_model(subj="subj1",pred="pred1",sig1=0.1,sig2=0.1,l1_sig1=10.0,resolution=(100,0.1),quds=['qud2'],only_trivial=False,just_s1=False,just_l0=False)
    # l1_model(subj="subj1",pred="pred2",sig1=0.1,sig2=0.1,l1_sig1=10.0,resolution=(100,0.1),quds=['qud2'],only_trivial=False,just_s1=False,just_l0=False)
