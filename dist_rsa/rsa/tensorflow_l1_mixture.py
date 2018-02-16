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
    double_tensor_projection_matrix,combine_quds, lookup, s1_nonvect,double_tensor_projection_matrix_into_subspace
from edward.inferences import HMC
import itertools
from dist_rsa.rsa.tensorflow_s1 import tf_s1

def tf_l1(inference_params):

	# NUM_QUDS: len(qud_combinations)
	# NUM_UTTS = number_of_utts


	#ALL THE SETUP STUFF
	if not inference_params.variational:
		raise Exception("Only Variational is Implemented")
	listener_world = tf.cast(inference_params.subject_vector,dtype=tf.float32)
	number_of_qud_words = len(inference_params.quds)
	poss_utts = tf.cast(as_a_matrix(inference_params.possible_utterances,inference_params.vecs),dtype=tf.float32)
	number_of_utts = len(inference_params.possible_utterances)
	poss_utt_freqs = np.log(np.array([lookup(inference_params.poss_utt_frequencies,x,'NOUN') for x in inference_params.possible_utterances]))
	weighted_utt_frequency_array = poss_utt_freqs-scipy.misc.logsumexp(poss_utt_freqs)
	qud_weight = tf.cast(inference_params.qud_weight,dtype=tf.float32)

	if not inference_params.qud_frequencies:
		qud_frequencies = inference_params.poss_utt_frequencies
	qud_freqs = -np.log(np.array([lookup(inference_params.qud_frequencies,x,'ADJ') for x in inference_params.quds]))
	if inference_params.trivial_qud_prior:
		weighted_qud_frequency_array = tf.expand_dims(tf.zeros([number_of_qud_words])+tf.log(1/number_of_qud_words),1)
	else: weighted_qud_frequency_array = tf.cast(tf.expand_dims(qud_freqs-scipy.misc.logsumexp(qud_freqs),1),dtype=tf.float32)
	qud_combinations = combine_quds(inference_params.quds,inference_params.number_of_qud_dimensions)
	print("qud_combinations",len(qud_combinations))
	print("quds",len(inference_params.quds))
	# SHAPE: [NUM_QUDS, NUM_DIMS, 1]
	qud_matrix = tf.cast((np.asarray([np.asarray([inference_params.vecs[word] for word in words]).T for words in qud_combinations])),dtype=tf.float32)
	inference_params.weighted_utt_frequency_array=weighted_utt_frequency_array
	inference_params.poss_utts=poss_utts
	inference_params.qud_matrix=qud_matrix
	inference_params.listener_world=listener_world 
	u=inference_params.predicate
	utt = tf.cast(inference_params.possible_utterances.index(u),dtype=tf.int32)

	# SHAPE: [NUM_QUDS,NUM_DIMS,NUM_QUD_DIMS]
	qud_projection_matrix = double_tensor_projection_matrix_into_subspace(inference_params.qud_matrix)
	# SHAPE: [NUM_QUDS,1,NUM_QUD_DIMS]
	projected_listener_worlds = tf.transpose(tf.einsum('aij,ik->akj',qud_projection_matrix,tf.expand_dims(listener_world,1)),perm=[0,2,1])

	# THE CORE L1 CODE STARTS HERE
	modes = []
	inferred_world_means = []
	sess = ed.get_session()
	for qi in range(len(qud_combinations)):

		# SHAPE: scalar
		projected_listener_world = tf.squeeze(projected_listener_worlds[qi])
		# SHAPE: [1]
		world = Normal(loc=projected_listener_world, scale=[inference_params.l1_sig1] * inference_params.number_of_qud_dimensions)
		# FEED S1 the world sample in the dimension of the original space
		# SHAPE: [NUM_QUDS,NUM_UTTS]
		l = tf_s1(inference_params,s1_world=tf.matmul(tf.expand_dims(world,0),tf.transpose(inference_params.qud_matrix[qi])))

		# SHAPE: scalar
		full_l = Categorical(logits=l[qi])

		# SHAPE: [1]
		qworld = Normal(loc=tf.Variable(projected_listener_world),scale=[0.1] * inference_params.number_of_qud_dimensions)

		init = tf.global_variables_initializer()
		inference_variational = ed.KLqp({world: qworld}, data={full_l: utt})
		optimizer = tf.train.RMSPropOptimizer(learning_rate=inference_params.step_size)
		inference_variational.initialize(optimizer=optimizer)
		tf.global_variables_initializer().run()
		inference_variational.run(n_iter=inference_params.variational_steps)
		modes.append(tf.matmul(tf.expand_dims(qworld.mode(),0),tf.transpose(inference_params.qud_matrix[qi])))
		inferred_world_means.append(qworld.mean())
	# SHAPE: [NUM_QUDS,NUM_DIMS]
	modes = tf.squeeze(tf.stack(modes))
	
	# SHAPE: [NUM_SAMPLES=NUM_QUDS,NUM_QUDS,NUM_UTTS]
	s1_outputs = tf.map_fn(lambda w: tf_s1(inference_params,s1_world=tf.expand_dims(w,0)),modes)

	# SHAPE: [NUM_SAMPLES,NUM_QUDS]
	s1_at_utt = s1_outputs[:,:,utt]
	# FOR EACH ROW: SUBTRACT SUM OF ROW FROM EACH ELEMENT OF ROW
	# SHAPE: [NUM_SAMPLES,NUM_QUDS]
	inferred_qud = tf.subtract(s1_at_utt,tf.expand_dims(tf.reduce_logsumexp(s1_at_utt,axis=-1),1))
	# TAKE MEAN ACROSS WORLD SAMPLES
	# SHAPE: [NUM_QUDS]
	inferred_qud = tf.subtract(tf.reduce_logsumexp(inferred_qud,axis=0),tf.log(tf.cast(tf.shape(inferred_qud)[0],dtype=tf.float32)))
	
	# PAIR QUD PROBABILITIES WITH THEIR QUDS AND SORT BY LARGEST PROB MASS
	results = list(zip(qud_combinations,sess.run(inferred_qud)))
	results = (sorted(results, key=lambda x: x[1], reverse=True))

	return results,sess.run(inferred_world_means)





