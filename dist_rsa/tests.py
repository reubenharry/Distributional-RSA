# from __future__ import division
# import nose
# # import scipy
# # from utils.load_data import *
# # from utils.helperfunctions import *
# # from utils.helperfunctions import visualize_cosine
# # import numpy as np
# # import pickle
# # import itertools
# # from utils.refine_vectors import h_dict,processVecMatrix
# # import nltk
# # import glob
import pickle
from dist_rsa.rsa.numpy_s1 import np_rsa
from dist_rsa.utils.load_data import get_freqs
# from dist_rsa.rsa.tensorflow_s1 import tf_s1
# from dist_rsa.rsa.tensorflow_s1_old import tf_s1_old
# from dist_rsa.rsa.tensorflow_s1_triple_vec import tf_s1_triple_vec as tf_s1_newest
from dist_rsa.utils.load_data import load_vecs,get_freqs
import edward as ed
from dist_rsa.dbm import *

# frequencies=get_freqs(preprocess=False)
vec_length=25
vecs = load_vecs(mean=True,pca=True,vec_length=vec_length,vec_type="glove.twitter.27B.")    
sig1=1.0
sig2=1.0
world = 'human'
utt = vecs['shark']
quds = ['vicious','wet','clever']
possible_utterances = ['man','shark']

inference_params = Inference_Params(
    vecs=vecs,
    subject=[world],predicate=utt,
    quds=quds,
    possible_utterances=possible_utterances,
    sig1=sig1,sig2=sig2,l1_sig1=0.1,
    qud_weight=0.0,freq_weight=0.0,
    categorical="categorical",
    sample_number = 500,
    number_of_qud_dimensions=2,
    burn_in=400,
    seed=False,trivial_qud_prior=False,
    step_size=0.0005,
    poss_utt_frequencies=defaultdict(lambda:1),
    qud_frequencies=defaultdict(lambda:1),
    qud_prior_weight=0.5,
    rationality=1.0,
    norm_vectors=False
    )



s1_world = inference_params.vecs[world]+inference_params.vecs["the"]

s1_world_2 = inference_params.vecs[world]+inference_params.vecs["a"]


# run_old = Dist_RSA_Inference(inference_params)
# run_old.compute_s1(s1_world=np.expand_dims(s1_world,0),debug=False,vectorization=1)

run_new = Dist_RSA_Inference(inference_params)
run_new.compute_s1(s1_world=np.expand_dims(s1_world,0),debug=True,vectorization=2)

# run_newest = Dist_RSA_Inference(inference_params)
# run_newest.compute_s1(s1_world=np.array([s1_world,s1_world_2]),debug=False,vectorization=3)



# tf_old = ed.get_session().run(run_old.s1_results)
tf_new = ed.get_session().run(run_new.s1_results)
# tf_newest = ed.get_session().run(run_newest.s1_results)

print("listener mean",vecs[world])

print("tf new",tf_new)

# print(tf_old,'\n\n\n\n\n',tf_new,"\n\n\n\n\n",tf_newest[0])
# print(tf_old.shape,tf_new.shape,tf_newest.shape)
# print(tf_newest[0]==tf_old)
# print("ARE NEW AND OLD TF EQUAL?",np.array_equal(tf_old,tf_new),np.array_equal(tf_old,tf_newest[0]))

# tf_new = ed.get_session().run(tf_s1(inference_params,s1_world))
# tf_old = ed.get_session().run(tf_s1_old(inference_params,s1_world))


np_unvectorized = np.array(np_rsa(s1_world=s1_world, qud_mat=np.array([vecs['vicious'],vecs['wet']]).T, vecs=vecs,vec_length=vec_length, listener_prior_mean=vecs[world], possible_utterances=possible_utterances, utterance=utt,sig1=sig1,sig2=sig2,frequencies=defaultdict(lambda:1)))

print("numpy",np_unvectorized)


# print(np_unvectorized)
# for qud in quds:
#     for world in [s1_world,s1_world_2]:

# qud_combinations...

#         np_unvectorized = np.array(np_rsa(s1_world=world, qud_mat=np.array([vecs[x]]).T, vecs=vecs,vec_length=vec_length, world_mean=world, possible_utterances=possible_utterances, utterance=utt,sig1=sig1,sig2=sig2,frequencies=defaultdict(lambda:1)))
#         print(np_unvectorized)

# t = (ed.get_session().run(run.s1_results))
# print(np_unvectorized)
# print(tf_newest.shape,tf_new.shape,tf_old.shape,np_unvectorized.shape)

# print(tf_newest,tf_new,tf_old,np_unvectorized)
# assert(tf_new[0][0][0] == tf_old[0][0] == np_unvectorized)