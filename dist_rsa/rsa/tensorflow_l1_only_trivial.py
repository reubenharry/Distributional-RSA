def tf_l1_only_trivial(inference_params):

	from dist_rsa.rsa.tensorflow_s1 import tf_s1
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
	from dist_rsa.rsa.tensorflow_s1_trivial import tf_s1_trivial

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
	world = Normal(loc=tf.squeeze(listener_world), scale=[inference_params.sig1] * inference_params.vec_length)

	l = tf_s1_trivial(inference_params,s1_world=world)
	l_summed = tf.reduce_logsumexp(l,axis=0)

	print("length poss utts and l_summed",len(inference_params.possible_utterances),l_summed.shape)

	# zipped_lists = sorted(list(zip(np.ndarray.tolist(l_summed),possible_utterances)),key = lambda x : x[0],reverse=True)[:crop]

	# unzipped_lists = zip(*zipped_lists)

	# u = tf.cast(unzipped_lists[1].index(u),dtype=tf.int32)

	full_l = Categorical(logits=l_summed)
	# full_l = Categorical(logits=unzipped_lists[0])

	# print("SHAPES OF FULLL AND U",full_l.shape,u.shape)





	if inference_params.variational:
		qworld = Normal(loc=tf.Variable(tf.zeros(inference_params.vec_length)),scale=tf.exp(tf.Variable(tf.ones(inference_params.vec_length))))
		init = tf.global_variables_initializer()
		inference_variational = ed.KLqp({world: qworld}, data={full_l: utt})
		optimizer = tf.train.RMSPropOptimizer(1e-3, epsilon=1.0)
		inference_variational.initialize(optimizer=optimizer)
		tf.global_variables_initializer().run()
		inference_variational.run(n_iter=inference_params.variational_steps)
		inferred_world = qworld.sample(sample_shape=(inference_params.sample_number))
	else:
		qworld = Empirical(params=tf.Variable(tf.random_normal([inference_params.sample_number, inference_params.vec_length])))
		inference = ed.HMC({world: qworld},data={full_l: utt})
		inference.run(step_size=inference_params.step_size)
		inferred_world = qworld.params


	sess = ed.get_session()
	return sess.run(inferred_world)

