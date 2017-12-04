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
import edward as ed
import random

random.seed(9001)


vecs = load_vecs(mean=True,pca=True,vec_length=300,vec_type='glove.6B.')
nouns,adjs = get_words()

freqs = pickle.load(open('dist_rsa/data/google_freqs/freqs','rb'))
nouns,adjs = get_words(with_freqs=False)

qud_words = [a for a in list(adjs) if adjs[a] < abstract_threshold and a in vecs]



qud_words = sorted(qud_words,\
#     key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[pred]],axis=0)),reverse=False)
    key=lambda x:freqs[x],reverse=True)

noun_words = [n for n in nouns if nouns[n] > concrete_threshold and n in vecs]
noun_words = sorted(noun_words,\
    key=lambda x:freqs[x],reverse=True)

# print("QUDS AND NOUNS 1",qud_words[0],noun_words[0])

random.shuffle(qud_words)
random.shuffle(noun_words)

def l1_model(metaphor):
    vec_size,vec_kind = 300,'glove.840B.'
    subj,pred,sig1,sig2,l1_sig1,start,stop,is_baseline,qud_num = metaphor

    print('abstract_threshold',abstract_threshold)
    print('concrete_threshold',concrete_threshold)

    # quds = ['commanding','ruthless']
    # possible_utterances = ["puppy","child","principal"]

    quds = qud_words[:500]+[subj]
    possible_utterances = noun_words[:500]+[subj]
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
        possible_utterances=sorted(list(set(possible_utterances).union(set([pred])))),
        sig1=sig1,sig2=sig2, l1_sig1=l1_sig1,
        qud_weight=0.0,freq_weight=0.0,
        categorical="categorical",
        sample_number = 1000,
        number_of_qud_dimensions=1,
        # burn_in=900,
        seed=False,trivial_qud_prior=False,
        step_size=1e-10,
        poss_utt_frequencies=defaultdict(lambda:1),
        qud_frequencies=defaultdict(lambda:1),
        qud_prior_weight=0.5,
        rationality=0.99,
        norm_vectors=False,
        variational=True,
        variational_steps=100,
        baseline=is_baseline
        # world_movement=True

        )

    run = Dist_RSA_Inference(params)

    run.compute_l1(load=0,save=False)


    l0_post = tf.transpose(tf.divide(tf.add(run.inference_params.listener_world/run.inference_params.sigma1, 
      run.inference_params.poss_utts/run.inference_params.sigma2),run.inference_params.inverse_sd))

    l0_post = ed.get_session().run(l0_post)

    print("L0 post shape",l0_post.shape)

    for q in ["strong","fast","green"]:
        projected_l0_post = projection_debug(l0_post,np.expand_dims(real_vecs[q],-1))
        subj_proj = projection_debug(run.inference_params.subject_vector,np.expand_dims(real_vecs[q],-1))
        # print(run.inference_params.subject_vector)
        print(q,projected_l0_post,subj_proj)

    print("projection of man on green",projection_debug(real_vecs["man"],np.expand_dims(real_vecs["green"],1)))
    print("projection of horse on green",projection_debug(real_vecs["horse"],np.expand_dims(real_vecs["green"],1)))
    print("projection of oak on green",projection_debug(real_vecs["oak"],np.expand_dims(real_vecs["green"],1)))

    # print("man",real_vecs["man"])
    # print("horse",real_vecs["horse"])

    # qud_projection_matrix = double_tensor_projection_matrix(run.inference_params.qud_matrix)
    # projected_mus = tf.transpose(tf.einsum('aij,jk->aik',qud_projection_matrix,l0_post),perm=[0,2,1])
    # projected_world = tf.transpose(tf.einsum('aij,jk->aik',qud_projection_matrix,tf.expand_dims(run.inference_params.listener_world,1)),perm=[0,2,1])

    # print(ed.get_session().run([projected_world-projected_mus]))

    print("VECTOR BASELINE",sorted(quds,\
        key=lambda x:scipy.spatial.distance.cosine(vecs[x],np.mean([vecs[subj],vecs[pred]],axis=0)),reverse=False))

    results = run.qud_results()
    if not is_baseline:
        # worldm = run.world_movement("cosine",comparanda=[x for x in quds if x in real_vecs])    
        worldm = run.world_movement("cosine",comparanda=[x for x in quds if x in real_vecs],do_projection=True)
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


    print(new_results)
    print(worldm)

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

