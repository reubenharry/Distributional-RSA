#calculates the quds directly from the world mean and demarginalization
def tf_l1_qud_only(inference_params):

	from dist_rsa.rsa.tensorflow_s1 import tf_s1
	import numpy as np
	import edward as ed
	import scipy
	from collections import Counter
	import pickle
	import tensorflow as tf
	from edward.models import Normal,Empirical, Bernoulli, Categorical
	from dist_rsa.utils.helperfunctions import projection,tensor_projection,weights_to_dist,\
        normalize,as_a_matrix,tensor_projection_matrix,\
        double_tensor_projection_matrix,combine_quds, lookup, s1_nonvect
	from edward.inferences import HMC
	import itertools

	listener_world = tf.cast(inference_params.subject_vector,dtype=tf.float32)
	number_of_quds = len(inference_params.quds)
	poss_utts = tf.cast(as_a_matrix(inference_params.possible_utterances,inference_params.vecs),dtype=tf.float32)
	number_of_utts = len(inference_params.possible_utterances)
	poss_utt_freqs = np.log(np.array([lookup(inference_params.poss_utt_frequencies,x,'NOUN') for x in inference_params.possible_utterances]))
	weighted_utt_frequency_array = poss_utt_freqs-scipy.misc.logsumexp(poss_utt_freqs)
	

	qud_weight = tf.cast(inference_params.qud_weight,dtype=tf.float32)

	u=inference_params.predicate
	utt = tf.cast(inference_params.possible_utterances.index(u),dtype=tf.int32)

	# if not inference_params.qud_frequencies:
	qud_frequencies = inference_params.poss_utt_frequencies

	qud_freqs = -np.log(np.array([lookup(inference_params.qud_frequencies,x,'ADJ') for x in inference_params.quds]))
	# if inference_params.trivial_qud_prior:
	# 	weighted_qud_frequency_array = tf.expand_dims(tf.zeros([number_of_quds])+tf.log(1/number_of_quds),1)
	# else:
	weighted_qud_frequency_array = tf.cast(tf.expand_dims(qud_freqs-scipy.misc.logsumexp(qud_freqs),1),dtype=tf.float32)
	
	#remove to save worry about failing: needed for multidim categorical
	qud_combinations = combine_quds(inference_params.quds,inference_params.number_of_qud_dimensions)
	# qud_combinations = [[q] for q in quds]

	print("qud_combinations",len(qud_combinations))
	print("quds",len(inference_params.quds))

	qud_matrix = tf.cast((np.asarray([np.asarray([inference_params.vecs[word] for word in words]).T for words in qud_combinations])),dtype=tf.float32)

	inference_params.weighted_utt_frequency_array=weighted_utt_frequency_array
	inference_params.poss_utts=poss_utts
	inference_params.qud_matrix=qud_matrix
	inference_params.listener_world=listener_world 

		#calculates the variance of the 2 normals involved in the L0


	inferred_qud = tf.expand_dims(tf_s1(inference_params,s1_world=tf.expand_dims(listener_world,0)),0)
	# tf.concat([out1,out2],axis=0)
	# inferred_qud = inferred_qud[inference_params.burn_in:]



	inferred_qud = inferred_qud[:,:,utt]
	inferred_qud = tf.subtract(inferred_qud,tf.expand_dims(tf.reduce_logsumexp(inferred_qud,axis=-1),1))


	inferred_qud = tf.subtract(tf.reduce_logsumexp(inferred_qud,axis=0),tf.log(tf.cast(tf.shape(inferred_qud)[0],dtype=tf.float32)))

	sess = tf.Session()

	results = list(zip(qud_combinations,sess.run(inferred_qud)))
	results = (sorted(results, key=lambda x: x[1], reverse=True))
	inference_params.results=results

	return results



