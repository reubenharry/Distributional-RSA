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
    double_tensor_projection_matrix,combine_quds, lookup,\
    s1_nonvect,double_tensor_projection_matrix_into_subspace,\
    orthogonal_complement_tf,projection_into_subspace_tf, projection_into_subspace_np, orthogonal_complement_np
from edward.inferences import HMC
import itertools
from dist_rsa.rsa.tensorflow_s1 import tf_s1

def tf_l1(inference_params):

	# NUM_QUDS: len(qud_combinations)
	# NUM_UTTS = number_of_utts
	NUM_DIMS = inference_params.subject_vector.shape[0]
	NUM_QUD_DIMS = inference_params.number_of_qud_dimensions


	#ALL THE SETUP STUFF
	sess = ed.get_session()
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
	# SHAPE: [NUM_QUDS, NUM_DIMS, NUM_QUD_DIMS]
	qud_matrix = tf.cast((np.asarray([np.asarray([inference_params.vecs[word]/np.linalg.norm(inference_params.vecs[word]) for word in words]).T for words in qud_combinations])),dtype=tf.float32)
	print("QUD MATRIX",qud_matrix,sess.run(qud_matrix))
	# print(qud_combinations)
	inference_params.weighted_utt_frequency_array=weighted_utt_frequency_array
	inference_params.poss_utts=poss_utts
	inference_params.qud_matrix=qud_matrix
	inference_params.listener_world=listener_world 
	u=inference_params.predicate
	utt = tf.cast(inference_params.possible_utterances.index(u),dtype=tf.int32)

	# SHAPE: [NUM_QUDS,NUM_DIMS,NUM_QUD_DIMS]
	qud_projection_matrix = double_tensor_projection_matrix_into_subspace(inference_params.qud_matrix)
	# SHAPE: [NUM_QUDS,1,NUM_QUD_DIMS]
	# CHANGE FOR 2D CASE
	print("projection shapes",tf.expand_dims(listener_world,1),tf.transpose(tf.squeeze(qud_matrix)))
	projected_listener_worlds = projection_into_subspace_tf(tf.expand_dims(listener_world,1),tf.transpose(tf.squeeze(qud_matrix)))

	# print("shape qud matrix squeezed",tf.squeeze(qud_matrix))
	print("shape projected listener worlds",projected_listener_worlds)

	pi = tf.constant(np.pi)
	
# tf.zeros([NUM_DIMS-NUM_QUD_DIMS,1])
	# tf.transpose(tf.einsum('aij,ik->akj',qud_projection_matrix,tf.expand_dims(listener_world,1)),perm=[0,2,1])


	"""
	functions:

		get sample from full space: input: mean and variance of a given qud takes samples from qud and each orthogonal subspace and unprojects and sums

		reconstruct_qud_probs(): given samples of worlds from each qud, reconstructs weights of each qud in mixture model



	"""

	determinants=[]
	means=[]

	#ALL THIS CODE IS WRONG FOR MULTIDIMENSIONAL QUDS

	full_space_prior = Normal(loc=tf.squeeze(listener_world), scale=[inference_params.l1_sig1] * inference_params.vec_length)


	combined_samples = []
	for qi in range(len(qud_combinations)):

		#CALCULATE MEAN AND SIGMA OF p(w|q=q,u=u) in the projected space; currently with VI

		# SHAPE: scalar
		projected_listener_world = projected_listener_worlds[qi]
		# print("shape of projected listener world",projected_listener_world)
		# SHAPE: [1]
		world = Normal(loc=projected_listener_world, scale=[inference_params.l1_sig1] * inference_params.number_of_qud_dimensions)
		# FEED S1 the world sample in the dimension of the original space
		# SHAPE: [NUM_QUDS,NUM_UTTS]

		# print("world shape",world)
		# print("input shape", world*tf.transpose(inference_params.qud_matrix[qi]))
		# NEEDS changing for 2d
		l = tf_s1(inference_params,s1_world=world*tf.transpose(inference_params.qud_matrix[qi]))
			# tf.matmul(tf.expand_dims(world,0),tf.transpose(inference_params.qud_matrix[qi])))
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
		
		#CALCULATE FULL MEAN AND COVARIANCE FOR A SINGLE GAUSSIAN:

	

		# SHAPE of inference_params.qud_matrix[qi] : [NUM_DIMS, NUM_QUD_DIMS]

		full_basis_qud_mean = tf.transpose(qworld.mean()*inference_params.qud_matrix[qi])
			#multidim version
			# tf.matmul(qworld.mean(),tf.transpose(inference_params.qud_matrix[qi]))
		# print("full_basis_qud_mean",full_basis_qud_mean)


		#could do this whole bit in numpy
		# THE ORTHOGONAL VECTORS TO CURRENT QUD Q
		# SHAPE: [NUM_DIMS,NUM_DIMS-NUM_QUD_DIMS]
		# current_qud = qud_matrix[qi] - tf.norm(qud_matrix[qi])
		# change for 2d
		# a = sess.run(tf.transpose(qud_matrix[qi]))
		# print(a,orthogonal_complement_np(a),orthogonal_complement_np(a.T))
		orthogonal_dims = tf.transpose(orthogonal_complement_tf(qud_matrix[qi]))
		# print("shapes for proj",orthogonal_dims)

		orthogonal_basis = tf.concat([orthogonal_dims,tf.transpose(qud_matrix[qi])],axis=0)
		print("orthogonal_basis",orthogonal_basis)
		print("qr check",np.identity(NUM_DIMS),sess.run(tf.matmul(tf.transpose(orthogonal_basis),orthogonal_basis)))
		
		# raise Exception
		# print("qr check",orthogonal_dims,qud_matrix[qi])
		# print("projection shapes 2",tf.expand_dims(listener_world,1),tf.transpose(orthogonal_dims))
		projected_orthogonal_means = projection_into_subspace_tf(tf.expand_dims(listener_world,1),tf.transpose(orthogonal_dims))

		print("full mean shapes",projected_orthogonal_means,tf.transpose(qud_matrix[qi]))
		# PROJECT THE LISTENER WORLD (e.g. man in "man is shark") into each orthogonal_dim: 
		# SHAPE: [1, NUM_DIMS-NUM_QUD_DIMS, NUM_QUD_DIMS]
			# WOULD BE GOOD TO DOUBLE CHECK THIS!!
		# print("projected_orthogonal_means",projected_orthogonal_means)
		# ? WHAT IS THE OPERATION TO UNPROJECT THE PROJECTED MEANS
		# DO ELEMENTWISE MULTIPLICATION WITH ORTHONGAL DIMS
		full_basis_orthogonal_means = projected_orthogonal_means*tf.transpose(qud_matrix[qi])
			#multidim version
		# print("full_basis_orthogonal_means",full_basis_orthogonal_means)
		concat_means = tf.concat([full_basis_orthogonal_means,full_basis_qud_mean],axis=0)
		full_mean = tf.reduce_sum(concat_means,axis=0)
		# print("full mean shape",full_mean)
		means.append(full_mean)

		# WHAT ORDER SHOULD THESE BE IN?
		covariance = tf.concat([tf.zeros([NUM_DIMS-NUM_QUD_DIMS])+inference_params.l1_sig1,qworld.variance()],axis=0)
		
		two_pi_cov = 2*pi*covariance
		# print(covariance,"covariance")
		determinant = tf.matrix_determinant(tf.diag(covariance))

		determinants.append(determinant)



		# m = means[qi]

	def posterior_prob(w,qi):



		det = determinants[qi]

		term_1 = tf.log(tf.sqrt(det))

		# print(w)
		# print(tf.expand_dims(w))

		term_2 = tf_s1(inference_params,s1_world=tf.expand_dims(w,0))[qi][utt]
		term_3 = tf.reduce_prod(full_space_prior.log_prob(w))

		# print(term_1,term_2,term_3, "terms")

		return term_1+term_2+term_3




	# print("test func", posterior_prob(means[0],0))

	qud_scores = tf.stack([posterior_prob(means[qi],qi) for qi in range(len(qud_combinations))])

	qud_distribution = qud_scores - tf.reduce_logsumexp(qud_scores,axis=0)

	def world_sampler():
		# pass
		unit_gaussian = Normal(loc=0.0,scale=1.0)
		full_basis_qud_sample = tf.transpose(qworld.sample()*inference_params.qud_matrix[qi])
		orthogonal_dims = tf.transpose(orthogonal_complement_tf(qud_matrix[qi]))
		projected_orthogonal_means = projection_into_subspace_tf(tf.expand_dims(listener_world,1),tf.transpose(orthogonal_dims))
		samples = unit_gaussian.sample(NUM_DIMS-NUM_QUD_DIMS,1)
		projected_orthogonal_samples = projected_orthogonal_means+(samples*inference_params.l1_sig1)
		print("full mean shapes",projected_orthogonal_means,tf.transpose(qud_matrix[qi]))
		full_basis_orthogonal_samples = projected_orthogonal_means*tf.transpose(qud_matrix[qi])
			#multidim version
		# print("full_basis_orthogonal_means",full_basis_orthogonal_means)
		concat_samples = tf.concat([full_basis_orthogonal_samples,full_basis_qud_sample],axis=0)
		full_samples = tf.reduce_sum(concat_samples,axis=0) 

		return full_samples

	# def world_sampler():
	# 	# pass
	# 	unit_gaussian = np.random.normal(loc=np.zeros([NUM_DIMS-NUM_QUD_DIMS,1]),scale=1.0)
	# 	full_basis_qud_sample = np.transpose(np.random.normal(loc=sess.run(qworld.mean()*inference_params.qud_matrix[qi]),scale=sess.run(qworld.variance())))
	# 	orthogonal_dims = sess.run(tf.transpose(orthogonal_complement_tf(qud_matrix[qi])))
	# 	projected_orthogonal_means = projection_into_subspace_np(np.expand_dims(inference_params.subject_vector,1),np.transpose(orthogonal_dims))
	# 	samples = unit_gaussian
	# 	projected_orthogonal_samples = projected_orthogonal_means+(samples*inference_params.l1_sig1)
	# 	# print("full mean shapes",projected_orthogonal_means,np.transpose(qud_matrix[qi]))
	# 	full_basis_orthogonal_samples = projected_orthogonal_means*sess.run(tf.transpose(qud_matrix[qi]))
	# 		#multidim version
	# 	# print("full_basis_orthogonal_means",full_basis_orthogonal_means)
	# 	concat_samples = np.concatenate([full_basis_orthogonal_samples,full_basis_qud_sample],axis=0)
	# 	full_samples = np.sum(concat_samples,axis=0) 

	# 	return full_samples

	# world_samples = sess.run([world_sampler() for _ in range(inference_params.sample_number)])
	world_samples=None

	# print(qud_distribution,np.exp(sess.run(qud_distribution)))

	# raise Exception
	results = list(zip(qud_combinations,sess.run(qud_distribution)))
	results = (sorted(results, key=lambda x: x[1], reverse=True))

	print([(x,np.exp(y)) for (x,y) in results])

	print(world_samples)
	# raise Exception

	return results,world_samples



