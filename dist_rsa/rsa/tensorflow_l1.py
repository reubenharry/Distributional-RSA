def tf_l1(inference_params):

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
	from edward.inferences import HMC
	import itertools
	from dist_rsa.rsa.tensorflow_s1 import tf_s1
	# from dist_rsa.rsa.tensorflow_s1_triple_vec import tf_s1_triple_vec as tf_s1
	# from tensorflow_l0_sigma import tf_l0_sigma


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

		#calculates the variance of the 2 normals involved in the L0






	u=inference_params.predicate
	utt = tf.cast(inference_params.possible_utterances.index(u),dtype=tf.int32)
	# world = Normal(loc=tf.squeeze(listener_world), scale=[inference_params.sig1] * inference_params.vec_length)
	world = Normal(loc=tf.squeeze(listener_world), scale=[tf.sqrt(inference_params.l1_sig1)] * inference_params.vec_length)
	l = tf_s1(inference_params,s1_world=tf.expand_dims(world,0))
	# l = tf.squeeze(l)
	# print("l shape",l.get_shape())
	# print('weighted_qud_frequency_array shape',weighted_qud_frequency_array.shape)
	# print(world.get_shape(),qud_matrixrix.get_shape())
	# if inference_params.number_of_qud_dimensions == 1:
	# 	l = tf.add(l,tf.multiply(qud_weight,weighted_qud_frequency_array))
	l_summed = tf.reduce_logsumexp(l,axis=0)

	print("length poss utts and l_summed",len(inference_params.possible_utterances),l_summed.shape)

	# zipped_lists = sorted(list(zip(np.ndarray.tolist(l_summed),possible_utterances)),key = lambda x : x[0],reverse=True)[:crop]

	# unzipped_lists = zip(*zipped_lists)

	# u = tf.cast(unzipped_lists[1].index(u),dtype=tf.int32)
	full_l = Categorical(logits=l_summed)
	# full_l = Categorical(logits=unzipped_lists[0])

	# print("SHAPES OF FULLL AND U",full_l.shape,u.shape)


	sess = ed.get_session()


	# GET L0 posterior on the utterance
	mus_new = tf.divide(tf.add(inference_params.listener_world/inference_params.sigma1, 
		inference_params.poss_utts/inference_params.sigma2),inference_params.inverse_sd)
	mu_new = mus_new[utt]
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
	if inference_params.variational:
		# qworld = Normal(loc=tf.Variable(tf.zeros(inference_params.vec_length)),scale=tf.exp(tf.Variable(tf.ones(inference_params.vec_length))))
		qworld = Normal(loc=tf.Variable(tf.squeeze(listener_world)),scale=tf.Variable([tf.sqrt(inference_params.l1_sig1)] * inference_params.vec_length))
		# qworld = Normal(loc=tf.Variable(tf.squeeze(listener_world)),scale=tf.exp(tf.Variable(tf.ones(inference_params.vec_length))))
		# qworld = Normal(loc=tf.Variable(tf.squeeze(listener_world)),scale=[inference_params.sigma1] * inference_params.vec_length)
		# qworld = Normal(loc=tf.Variable(tf.squeeze(mu_new)),scale=[inference_params.sigma1] * inference_params.vec_length)
		print("QWORLD SHAPE", qworld.get_shape())
		init = tf.global_variables_initializer()
		inference_variational = ed.KLqp({world: qworld}, data={full_l: utt})
		optimizer = tf.train.RMSPropOptimizer(learning_rate=inference_params.step_size)
			# , epsilon=1e-08, decay=0.0)

		# optimizer = tf.train.RMSPropOptimizer(inference_params.step_size, epsilon=1.0)
		inference_variational.initialize(optimizer=optimizer)
			# optimizer=optimizer)
		tf.global_variables_initializer().run()
		inference_variational.run(n_iter=inference_params.variational_steps)
		inferred_world = qworld.sample(sample_shape=(inference_params.sample_number))
	else:
		qworld = Empirical(params=tf.Variable(tf.random_normal([inference_params.sample_number, inference_params.vec_length])))
		inference = ed.HMC({world: qworld},data={full_l: utt})
		inference.run(step_size=inference_params.step_size)
		inferred_world = qworld.params

	tic = time.time()
	#for efficiency purposes, the large computation of the s1s for each of the samples is split into two vectorized computations
	#and the results are then concatenated
	# number_of_parts = 1000

	# splits = tf.convert_to_tensor(tf.split(inferred_world,[inference_params.sample_number//number_of_parts]*number_of_parts,axis=0))
	# print(splits.get_shape())
	# # inferred_qud = tf.map_fn(lambda x: tf_s1(inference_params,s1_world=x),[splits[0],splits[1]])
	# inferred_qud_inter = tf.map_fn(lambda w : tf_s1(inference_params,s1_world=w), splits)
	# inferred_qud_inter_shape = tf.shape(inferred_qud_inter)
	# inferred_qud = tf.reshape(inferred_qud_inter,[inferred_qud_inter_shape[0]*inferred_qud_inter_shape[1],inferred_qud_inter_shape[2],inferred_qud_inter_shape[3]])
	# print("inferred qud",inferred_qud)

	# 	inferred_qud = inferred_qud[inference_params.burn_in:]
	# 	inferred_world = inferred_world[inference_params.burn_in:]
	
	if not inference_params.variational:
		inferred_world = inferred_world[inference_params.burn_in:]
	# 	num = (inference_params.sample_number-inference_params.burn_in) // 2
	# 	first_half_worlds = inferred_world[:(inference_params.sample_number-inference_params.burn_in)//2]
	# 	second_half_worlds = inferred_world[(inference_params.sample_number-inference_params.burn_in)//2:]
	# else:

	first_half_worlds = inferred_world[:inference_params.sample_number//2]
	second_half_worlds = inferred_world[inference_params.sample_number//2:]
	
	out1 =  tf.map_fn(lambda w: tf_s1(inference_params,s1_world=tf.expand_dims(w,0)),
		first_half_worlds)
	out2 =tf.map_fn(lambda w: tf_s1(inference_params,s1_world=tf.expand_dims(w,0)),
		second_half_worlds)
	inferred_qud = tf.concat([out1,out2],axis=0)

	# out1 = tf_s1(inference_params,s1_world=first_half_worlds)
	# out2 = tf_s1(inference_params,s1_world=second_half_worlds)

	# inferred_qud = full_mat
	# inferred_qud = tf_s1(inference_params,s1_world=inferred_world)
	print("inferred_qud shape",inferred_qud.get_shape())
	toc = time.time()
	print("TIME OF DEMARGINALIZATION",toc-tic)
	# print(tac-toc)


	inferred_qud = inferred_qud[:,:,utt]
	inferred_qud = tf.subtract(inferred_qud,tf.expand_dims(tf.reduce_logsumexp(inferred_qud,axis=-1),1))
	# print(tec-tac)

	inferred_qud = tf.subtract(tf.reduce_logsumexp(inferred_qud,axis=0),tf.log(tf.cast(tf.shape(inferred_qud)[0],dtype=tf.float32)))
	# print(tuc-tec)
	results = list(zip(qud_combinations,sess.run(inferred_qud)))
	results = (sorted(results, key=lambda x: x[1], reverse=True))
	print( [(x,np.exp(y)) for (x,y) in results] )
	# if return_tf:
	# 	return inferred_qud, inferred_world
	# else:
	if not inference_params.variational:
		return results,sess.run(inferred_world)
	return results,sess.run(tf.expand_dims(qworld.mean(),0))





