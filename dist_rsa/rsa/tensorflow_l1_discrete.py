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
	# from edward.inferences import HMC
	import itertools
	from dist_rsa.rsa.tensorflow_s1 import tf_s1
	from dist_rsa.rsa.tensorflow_l0_sigma import tf_l0_sigma

	# from dist_rsa.rsa.tensorflow_s1_triple_vec import tf_s1_triple_vec as tf_s1
	# from tensorflow_l0_sigma import tf_l0_sigma
	sess = ed.get_session()


	listener_world = tf.cast(inference_params.subject_vector,dtype=tf.float32)
	number_of_quds = len(inference_params.quds)
	poss_utts = tf.cast(as_a_matrix(inference_params.possible_utterances,inference_params.vecs),dtype=tf.float32)
	number_of_utts = len(inference_params.possible_utterances)
	poss_utt_freqs = np.log(np.array([lookup(inference_params.poss_utt_frequencies,x,'NOUN') for x in inference_params.possible_utterances]))
	weighted_utt_frequency_array = poss_utt_freqs-scipy.misc.logsumexp(poss_utt_freqs)
	

	qud_weight = tf.cast(inference_params.qud_weight,dtype=tf.float32)

	if not inference_params.qud_frequencies:
		qud_frequencies = inference_params.poss_utt_frequencies

	qud_freqs = -np.log(np.array([lookup(inference_params.qud_frequencies,x,'ADJ') for x in inference_params.quds]))
	if inference_params.trivial_qud_prior:
		weighted_qud_frequency_array = tf.expand_dims(tf.zeros([number_of_quds])+tf.log(1/number_of_quds),1)
	else:
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




	u=inference_params.predicate
	utt = tf.cast(inference_params.possible_utterances.index(u),dtype=tf.int32)
	# world = Normal(loc=tf.squeeze(listener_world), scale=[inference_params.sig1] * inference_params.vec_length)
	world = Normal(loc=tf.squeeze(listener_world), scale=[inference_params.l1_sig1] * inference_params.vec_length)
	# l = tf_s1(inference_params,s1_world=tf.expand_dims(world,0))
	# l_summed = tf.reduce_logsumexp(l,axis=0)

	
	# full_l = Categorical(logits=l_summed)
	



	# GET L0 posterior on the utterance
	# mus_new = tf.divide(tf.add(inference_params.listener_world/inference_params.sigma1, 
	# 	inference_params.poss_utts/inference_params.sigma2),inference_params.inverse_sd)
	# mu_new = mus_new[utt]

	# print("squeezed mu shape",tf.squeeze(mu_new).get_shape())
	# print("sigma shape",inference_params.inverse_sd)
	# inference_params.sigma1,inference_params.sigma2,inference_params.inverse_sd,inference_params.sigma,inference_params.inverse_sigma = tf.l0_sigma(inference_params)

	# no_qud = s1_no_qud(speaker_world=world,freq_weight=freq_weight)
	# print("no qud shape",no_qud.get_shape())
	# no_qud = no_qud+tf.log(trivial_qud_prior)
	# weighted_l_summed = tf.log(1.0-trivial_qud_prior) + l_summed
	# weighted_no_qud = tf.squeeze(tf.log(trivial_qud_prior) + no_qud)
	# full_l = tf.reduce_logsumexp([weighted_l_summed,weighted_no_qud],axis=0)
	# full_l = Categorical(logits=full_l)

	size,amount = inference_params.resolution


	discrete_worlds = np.asarray([[y*amount,-x*amount] for (x,y) in itertools.product(range(-size,size),range(-size,size))],dtype=np.float32)

	# test = np.asarray([0 for (x,y) in itertools.product(range(-100,100),range(-100,100))],dtype=np.float32)
	# test[1100]=5

	if inference_params.just_l0:

		inference_params.sigma1,inference_params.sigma2,inference_params.inverse_sd,inference_params.sigma,inference_params.inverse_sigma = tf_l0_sigma(inference_params)

		mus_new = tf.divide(tf.add(inference_params.listener_world/inference_params.sigma1, 
		inference_params.poss_utts/inference_params.sigma2),inference_params.inverse_sd)
		# print(mus_new.get_shape(),listener_world.get_shape())
		# raise Exception
		l0_post = Normal(loc=tf.squeeze(mus_new[utt]), scale=[inference_params.sigma[utt,utt]] * inference_params.vec_length)

		# print(inference_params.sigma.get_shape())

		l0_worlds_post = tf.map_fn(lambda w: tf.reduce_sum(l0_post.log_prob(w)),
		discrete_worlds)

		return None,np.reshape(sess.run(l0_worlds_post),(size*2,size*2))

	print("discrete_worlds shape", discrete_worlds.shape)

	discrete_worlds_prior = tf.map_fn(lambda w: tf.reduce_sum(world.log_prob(w)),
		discrete_worlds)

	print(sess.run(world.log_prob(discrete_worlds[0])))

	print("discrete_worlds_prior.get_shape", discrete_worlds_prior.get_shape())

	
	# for (x,y) in itertools.product(range(100),range(100)):
		# print(world.get_shape())
		# print(world.prob([0.0,0.0]))
		# print(np.asarray([x/10,y/10],dtype=np.float32))
		# discrete_worlds_prior[x,y]=world.log_prob([x/10,y/10])


	
	inferred_worlds =  tf.map_fn(lambda w: tf_s1(inference_params,s1_world=tf.expand_dims(w,0)),
		discrete_worlds)

	if not inference_params.just_s1:
		inferred_worlds = inferred_worlds[:,:,utt]

		inferred_worlds = tf.reduce_logsumexp(inferred_worlds,axis=-1)

		inferred_worlds += discrete_worlds_prior
		# normalize
		inferred_worlds = inferred_worlds - tf.reduce_logsumexp(inferred_worlds)

	else: 
		if len(inference_params.quds)==1:
			print("\n\n\nQUDS\n\n\n",inference_params.quds)
			inferred_worlds = inferred_worlds[:,inference_params.target_qud,utt]
		else: raise Exception

	# inferred_worlds = tf.subtract(tf.reduce_logsumexp(inferred_worlds,axis=0),tf.log(tf.cast(tf.shape(inferred_worlds)[0],dtype=tf.float32)))
	# print(tuc-tec)
	# results = list(zip(qud_combinations,sess.run(inferred_qud)))
	# results = (sorted(results, key=lambda x: x[1], reverse=True))
	# if return_tf:
	# 	return inferred_qud, inferred_world
	# else:

	# return None,np.reshape(test,(200,200))

	# return None,np.exp(np.reshape(sess.run(discrete_worlds_prior),(200,200)))

	# inferred_worlds[1,1]=10
	print(discrete_worlds.shape)
	return (np.reshape(sess.run(inferred_worlds),(size*2,size*2)),
		discrete_worlds,
		np.reshape(sess.run(discrete_worlds_prior),(size*2,size*2)))





