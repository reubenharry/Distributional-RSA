def tf_l1_discrete(inference_params):

	import time
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
	import itertools
	from dist_rsa.rsa.tensorflow_s1 import tf_s1
	from dist_rsa.rsa.tensorflow_l0_sigma import tf_l0_sigma

	sess = ed.get_session()
	listener_world = tf.cast(inference_params.subject_vector,dtype=tf.float32)
	number_of_quds = len(inference_params.quds)
	poss_utts = tf.cast(as_a_matrix(inference_params.possible_utterances,inference_params.vecs),dtype=tf.float32)
	number_of_utts = len(inference_params.possible_utterances)
	poss_utt_freqs = np.log(np.array([lookup(inference_params.poss_utt_frequencies,x,'NOUN') for x in inference_params.possible_utterances]))
	weighted_utt_frequency_array = poss_utt_freqs-scipy.misc.logsumexp(poss_utt_freqs)
	qud_weight = tf.cast(inference_params.qud_weight,dtype=tf.float32)

	#remove to save worry about failing: needed for multidim categorical
	qud_combinations = combine_quds(inference_params.quds,inference_params.number_of_qud_dimensions)
	qud_matrix = tf.cast((np.asarray([np.asarray([inference_params.vecs[word] for word in words]).T for words in qud_combinations])),dtype=tf.float32)
	inference_params.weighted_utt_frequency_array=weighted_utt_frequency_array
	inference_params.poss_utts=poss_utts
	inference_params.qud_matrix=qud_matrix
	inference_params.listener_world=listener_world 

	u=inference_params.predicate
	utt = tf.cast(inference_params.possible_utterances.index(u),dtype=tf.int32)
	world = Normal(loc=tf.squeeze(listener_world), scale=[inference_params.l1_sig1] * inference_params.vec_length)

	size,amount = inference_params.resolution

	# shape: [(size*2)**2, 2] : for each of the world positions, an array of its position
	discrete_worlds = np.asarray([[y*amount,x*amount] for (x,y) in itertools.product(range(-size,size),range(-size,size))],dtype=np.float32)
	# shape: [(size*2)**2, 1] use the world gaussian to calculate the prior at each point.
	discrete_worlds_prior = tf.expand_dims(tf.map_fn(lambda w: tf.reduce_sum(world.log_prob(w)),
		discrete_worlds),1)
	
	# shape: [(size*2)**2, num_of_quds,num_of_utts]
	s1_scores =  tf.map_fn(lambda w: tf_s1(inference_params,s1_world=tf.expand_dims(w,0)),
		discrete_worlds)


	if inference_params.just_s1:
		return sess.run(tf.reshape(s1_scores[:,0,utt],(size*2,size*2))),None

	# shape: [(size*2)**2, num_of_quds]
	s1_scores = s1_scores[:,:,utt]

	# shape: [(size*2)**2, num_of_quds]
	l1_joint_posterior_unnormed = discrete_worlds_prior + s1_scores
	# shape: [(size*2)**2, num_of_quds]
	l1_joint_posterior_normed = l1_joint_posterior_unnormed - tf.reduce_logsumexp(l1_joint_posterior_unnormed)

	# shape: [(size*2)**2,]
	world_posterior = tf.reduce_logsumexp(l1_joint_posterior_normed,axis=1)
	# shape: [num_of_quds]
	qud_posterior = tf.reduce_logsumexp(l1_joint_posterior_normed,axis=0)

	return sess.run([tf.reshape(world_posterior,(size*2,size*2)),qud_posterior])





