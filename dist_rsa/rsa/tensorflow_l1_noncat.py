def tf_l1_noncat(inference_params):

	import numpy as np
	import edward as ed
	import scipy
	from collections import Counter
	import pickle
	import tensorflow as tf
	from edward.models import Normal,Empirical, Bernoulli, Categorical
	from dist_rsa.utils.load_data import initialize
	from dist_rsa.utils.helperfunctions import projection,tensor_projection,weights_to_dist,\
        normalize,as_a_matrix,tensor_projection_matrix,\
        double_tensor_projection_matrix,combine_quds, lookup, s1_nonvect
	from edward.inferences import HMC
	import itertools
	# from dist_rsa.rsa.tensorflow_s1 import tf_s1
	from dist_rsa.rsa.tensorflow_s1_noncat import tf_s1
	import time

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





# 		inference = ed.HMC({world: qworld,qud : qqud}, data={l : u})

	u=inference_params.predicate
	utt = tf.cast(inference_params.possible_utterances.index(u),dtype=tf.int32)
	world = Normal(loc=tf.squeeze(listener_world), scale=[inference_params.sig1] * inference_params.vec_length)
	qud = Normal(loc=tf.zeros((1,inference_params.vec_length,inference_params.number_of_qud_dimensions),dtype=tf.float32), scale=tf.ones((inference_params.vec_length,inference_params.number_of_qud_dimensions),dtype=tf.float32))
	l = tf_s1(inference_params,s1_world=world,qud=qud)
	# print("l shape",l.get_shape())
	# print('weighted_qud_frequency_array shape',weighted_qud_frequency_array.shape)
	# print(world.get_shape(),qud_matrixrix.get_shape())
	l_summed = tf.reduce_logsumexp(l,axis=0)

	# empirical_init = tf.concat([tf.zeros([sample_number,inference_params.vec_length,1],dtype=tf.float32),tf.ones([sample_number,inference_params.vec_length,1],dtype=tf.float32)],axis=-1)+0.001
# 		# print(empirical_init.get_shape())
	# qqud = Empirical(params=tf.Variable(tf.random_normal([inference_params.sample_number,inference_params.vec_length,inference_params.number_of_qud_dimensions],dtype=tf.float32)))

# 		S1 = s1(speaker_world=world,qud_mat=tf.expand_dims(qud,0),freq_weight=freq_weight)
# 		print(S1.shape,"S1 SHAPE")

# 		l = Categorical(logits=tf.squeeze(S1))

	# print("length poss utts and l_summed",len(inference_params.possible_utterances),l_summed.shape)

	# zipped_lists = sorted(list(zip(np.ndarray.tolist(l_summed),possible_utterances)),key = lambda x : x[0],reverse=True)[:crop]

	# unzipped_lists = zip(*zipped_lists)

	# u = tf.cast(unzipped_lists[1].index(u),dtype=tf.int32)
	full_l = Categorical(logits=l_summed)
	# full_l = Categorical(logits=unzipped_lists[0])

	# print("SHAPES OF FULLL AND U",full_l.shape,u.shape)


	sess = ed.get_session()


	# no_qud = s1_no_qud(speaker_world=world,freq_weight=freq_weight)
	# print("no qud shape",no_qud.get_shape())
	# no_qud = no_qud+tf.log(trivial_qud_prior)
	# weighted_l_summed = tf.log(1.0-trivial_qud_prior) + l_summed
	# weighted_no_qud = tf.squeeze(tf.log(trivial_qud_prior) + no_qud)
	# full_l = tf.reduce_logsumexp([weighted_l_summed,weighted_no_qud],axis=0)
	# full_l = Categorical(logits=full_l)
	if inference_params.variational:
		qworld = Normal(loc=tf.Variable(tf.zeros(inference_params.vec_length)),scale=tf.exp(tf.Variable(tf.zeros(inference_params.vec_length))))
		qqud = Normal(loc=tf.Variable(tf.zeros((1,inference_params.vec_length,inference_params.number_of_qud_dimensions))),scale=tf.exp(tf.Variable(tf.ones((inference_params.vec_length,inference_params.number_of_qud_dimensions)))))
		init = tf.global_variables_initializer()
		
		print("SHAPES",qqud.get_shape(),qud.get_shape(),world.get_shape(),qworld.get_shape())

		inference_variational = ed.KLqp({world: qworld,qud: qqud}, data={full_l: utt})
		optimizer = tf.train.RMSPropOptimizer(inference_params.step_size)
			# , epsilon=1.0)
		inference_variational.initialize(optimizer=optimizer)
		tf.global_variables_initializer().run()
		inference_variational.run(n_iter=inference_params.variational_steps)
		# inferred_world = qworld.sample(sample_shape=(inference_params.sample_number))
		# inferred_qud = qqud.sample(sample_shape=(inference_params.sample_number))
		inferred_world = qworld.mean()
		inferred_qud = qqud.mean()
	else:
		qworld = Empirical(params=tf.Variable(tf.random_normal([inference_params.sample_number, inference_params.vec_length])))
		inference = ed.HMC({world: qworld,qud: qqud},data={full_l: utt})
		inference.run(step_size=inference_params.step_size)
		inferred_world = qworld.params
		inferred_qud = qqud.params

	# 	return inferred_qud, inferred_world
	# else:
	return sess.run([inferred_qud, tf.expand_dims(inferred_world,0)])





		
# 		u = tf.cast(possible_utterances.index(u),dtype=tf.int32)

# 		world = Normal(loc=tf.squeeze(listener_world), scale=[sig1] * inference_params.vec_length)
# 		qworld = Empirical(params=tf.Variable(tf.random_normal([sample_number, inference_params.vec_length])))

# 		#a = ed.models.Uniform(0.0,tf.cast(np.pi,dtype=tf.float32))
# #		qa = Empirical(params=tf.Variable(tf.zeros([sample_number,])))
# #		queueewedee = tf.expand_dims(tf.stack([tf.cos(a),tf.sin(a)]),1)


# 		inference.run(step_size=step_size,n_steps=2)
# 		sess = ed.get_session()
# 		init = tf.global_variables_initializer()
# 		# return qqud.params[burn_in:],qworld.params[burn_in:]
# 		inferred_world = sess.run(qworld.params[burn_in:])
# 		inferred_qud = sess.run(qqud.params[burn_in:])
# 		# print(inferred_qud.shape,"rsa qud shape")
# 		return inferred_qud,inferred_world