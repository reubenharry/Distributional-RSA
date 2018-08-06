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
        double_tensor_projection_matrix,combine_quds, lookup, s1_nonvect,mean_and_variance_of_dist_array,\
        projection_into_subspace_tf
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
	print("QUDS:",qud_combinations)
	qud_matrix = tf.cast((np.asarray([np.asarray([inference_params.vecs[word] for word in words]).T for words in qud_combinations])),dtype=tf.float32)
	inference_params.weighted_utt_frequency_array=weighted_utt_frequency_array
	inference_params.poss_utts=poss_utts
	inference_params.qud_matrix=qud_matrix
	inference_params.listener_world=listener_world 

	u=inference_params.predicate
	utt = tf.cast(inference_params.possible_utterances.index(u),dtype=tf.int32)
	world = Normal(loc=tf.squeeze(listener_world), scale=[tf.sqrt(inference_params.l1_sig1)] * inference_params.vec_length)

	size,amount = inference_params.resolution.size, inference_params.resolution.amount
	# shape: [(size*2)**2, 2] : for each of the world positions, an array of its position
	discrete_worlds = np.asarray([[x*amount,y*amount] for (x,y) in itertools.product(range(-size,size+1),range(-size,size+1))],dtype=np.float32)
	# shape: [(size*2)**2, 1] use the world gaussian to calculate the prior at each point.
	discrete_worlds_prior = tf.expand_dims(tf.map_fn(lambda w: tf.reduce_sum(world.log_prob(w)),
		discrete_worlds),1)

	# unnorm_prior_mean, unnorm_prior_var = mean_and_variance_of_dist_array(probs=tf.exp(discrete_worlds_prior_unnorm),support=discrete_worlds)
	# print("unnorm prior",sess.run([unnorm_prior_mean,unnorm_prior_var]))



	# discrete_worlds_prior = tf.expand_dims(discrete_worlds_prior_unnorm-tf.reduce_logsumexp(discrete_worlds_prior_unnorm),1)
	
	# norm_prior_mean, norm_prior_var = mean_and_variance_of_dist_array(probs=tf.exp(discrete_worlds_prior),support=discrete_worlds)
	# print("norm prior",sess.run([norm_prior_mean,norm_prior_var]))
	# shape: [(size*2)**2, num_of_quds,num_of_utts]
	s1_scores =  tf.map_fn(lambda w: tf_s1(inference_params,s1_world=tf.expand_dims(w,0)),
		discrete_worlds)


	# if inference_params.just_s1:
	# 	assert s1_scores.get_shape()[1]==1
	# 	return sess.run(tf.reshape(s1_scores[:,0,utt],(size*2+1,size*2+1))),None

	# shape: [(size*2)**2, num_of_quds]
	s1_scores = s1_scores[:,:,utt]

	# shape: [(size*2)**2, num_of_quds] n.b.: this is a [number_of_worlds,1]+[number_of_worlds,num_of_quds] sum, involving broadcasting
	l1_joint_posterior_unnormed = discrete_worlds_prior + s1_scores


	l1_joint_posterior_normed = l1_joint_posterior_unnormed - tf.reduce_logsumexp(l1_joint_posterior_unnormed)
	# a = sess.run(tf.exp(l1_joint_posterior_normed))
	# mean,_ = sess.run(mean_and_variance_of_dist_array(probs=np.expand_dims(a[:,0],1),support=discrete_worlds))
	# # print("sum should be unit", np.sum(world_posterior_np))
	# print("cond MEAN 1",mean)
	# a[:,1]
	# mean,_ = sess.run(mean_and_variance_of_dist_array(probs=np.expand_dims(a[:,1],1),support=discrete_worlds))
	# # print("sum should be unit", np.sum(world_posterior_np))
	# print("cond MEAN 2",mean)
	# raise Exception
	# raise Exception
	world_posterior_np = sess.run(tf.exp(tf.reduce_logsumexp(l1_joint_posterior_normed,axis=1)))
	# shape: [(size*2)**2, num_of_quds]

	# shape: [(size*2)**2,]


	# SOME CODE TO LOOK AT THE PROJECTED VARIANCE AND MEAN: ASSUMES JUST A SINGLE QUD PRESENT
	# projected_worlds = projection_into_subspace_tf(tf.transpose(discrete_worlds),qud_matrix[0])
	# print("projected worlds shape",projected_worlds)

	# MEAN,VARIANCE = mean_and_variance_of_dist_array(probs=world_squash,support=projected_worlds)
	# print("MEAN AND VARIANCE",sess.run([MEAN,VARIANCE]))


	# print(sess.run([tf.exp(discrete_worlds_prior), world_squash,tf.constant(np.array([x*amount for x in range(-size,size)]),dtype=tf.float32)]))
	# world_squash_0 = tf.reduce_mean(tf.reshape(world_posterior,(size*2,size*2)),axis=0)
	# print("MEAN AND VARIANCE AXIS 0",sess.run([mean_and_variance_of_dist_array(world_squash_0,tf.constant(np.array([x*amount for x in range(-size,size)]),dtype=tf.float32))]))

	# world_squash_1 = tf.reduce_mean(tf.reshape(world_posterior,(size*2,size*2)),axis=1)
	# print("MEAN AND VARIANCE AXIS 1",sess.run([mean_and_variance_of_dist_array(world_squash_1,tf.constant(np.array([x*amount for x in range(-size,size)]),dtype=tf.float32))]))
	# shaped_post = np.reshape(world_posterior_np,newshape=(size*2+1,size*2+1))
	# print(np.ndarray.tolist(world_posterior_np))
	# print(shaped_post)
	# print(shaped_post[8][3])
	# raise Exception
	# print(discrete_worlds)
	# hm = (np.reshape(discrete_worlds,newshape=(size*2+1,size*2+1,2)))
	# print(hm)
	# print(hm[4][7])
	mean,_ = sess.run(mean_and_variance_of_dist_array(probs=np.expand_dims(world_posterior_np,1),support=discrete_worlds))
	print("sum should be unit", np.sum(world_posterior_np))
	print("MEAN",mean)
	
	# print("support", amount*np.arange(-size,(size+1)))
	# print("amount",amount)
	# print(np.sum(np.reshape(world_posterior_np,newshape=(size*2+1,size*2+1)),axis=0))
	# print("CHECK MEAN", np.sum(np.sum(np.reshape(world_posterior_np,newshape=(size*2+1,size*2+1)),axis=1)*-amount*np.arange(-size,(size+1))))
	# ,
		# tf.reduce_sum(tf.reduce_sum(world_posterior,axis=1)*-np.arange(-size,size+1))]) )
	# print(sess.run(tf.nn.moments(tf.constant(np.array([0.0,0.7])), axes=[0])))

	# shape: [num_of_quds]
	qud_posterior = tf.reduce_logsumexp(l1_joint_posterior_normed,axis=0)

	results = list(zip(qud_combinations,sess.run(qud_posterior)))
	results = (sorted(results, key=lambda x: x[1], reverse=True))

	inference_params.heatmap=np.reshape(world_posterior_np,newshape=(size*2+1,size*2+1))

	return inference_params.heatmap,[(x,np.exp(y)) for (x,y) in results]





