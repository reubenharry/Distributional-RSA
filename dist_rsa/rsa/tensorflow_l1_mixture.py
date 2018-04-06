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
    orthogonal_complement_tf,projection_into_subspace_tf, projection_into_subspace_np, orthogonal_complement_np,\
    mean_and_variance_of_dist_array
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
	NUM_QUDS = len(qud_combinations)
	print("qud_combinations",NUM_QUDS,qud_combinations)
	print("quds",len(inference_params.quds))
	# SHAPE: [NUM_QUDS, NUM_DIMS, NUM_QUD_DIMS]
	qud_matrix = tf.cast((np.asarray([np.asarray([inference_params.vecs[word]/np.linalg.norm(inference_params.vecs[word]) for word in words]).T for words in qud_combinations])),dtype=tf.float32)
	# print("QUD MATRIX",qud_matrix,sess.run(qud_matrix))
	#CHECK NORMALIZATION
	# for i in range(qud_matrix.get_shape()[0]):
	# 	print(tf.transpose(qud_matrix[i]),tf.Session().run(tf.norm(tf.transpose(qud_matrix[0]))),"stuff")
	# raise Exception
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
	# print("projection shapes",tf.expand_dims(listener_world,1),qud_matrix,qud_matrix[:,:,0],tf.transpose(tf.squeeze(qud_matrix)))
	projected_listener_worlds = projection_into_subspace_tf(tf.expand_dims(listener_world,1),tf.transpose(qud_matrix[:,:,0]))
	pi = tf.constant(np.pi)
	size,amount = inference_params.resolution
	print("SIZE,AMOUNT",size,amount)
	determinants=[]
	covariances=[]
	means=[]
	heatmaps = []
	std = tf.sqrt(inference_params.l1_sig1)

	discrete_worlds = tf.constant(np.asarray([[y*amount,x*amount] for (x,y) in itertools.product(range(-size,size+1),range(-size,size+1))],dtype=np.float32))

	print("discrete_worlds",discrete_worlds)

	# CALCULATE MEAN AND SIGMA OF p(w|q=qi,u=u) in the projected space
	#ALL THIS CODE IS WRONG FOR MULTIDIMENSIONAL QUDS
	full_space_prior = Normal(loc=tf.squeeze(listener_world), scale=[std] * inference_params.vec_length)
	for qi in range(len(qud_combinations)):
		
		print(qud_combinations[qi],"CURRENT QUD")
		projected_listener_world = projected_listener_worlds[qi]
		# world = Normal(loc=projected_listener_world, scale=[std] * inference_params.number_of_qud_dimensions)
				
		# the discrete support: tf.stack turns list of vectors (each a point lying along qud) into tensor where each row is vector
		# shape: [num_worlds,num_qud_dims,num_dims] (n.b. num_worlds = size*2+1)
		discrete_worlds_along_qud = tf.stack([tf.transpose(inference_params.qud_matrix[qi])*amount*x for x in range(-size,size+1)])
		# print("shapes",discrete_worlds_along_qud,sess.run(discrete_worlds_along_qud[0]))

		# shape: [num_worlds]
		discrete_worlds_along_qud_prior = tf.map_fn(lambda w: tf.reduce_sum(full_space_prior.log_prob(w)),discrete_worlds_along_qud)

		# shape: [num_worlds,num_quds,num_utts]
		s1_scores =  tf.map_fn(lambda w: tf_s1(inference_params,s1_world=w),discrete_worlds_along_qud)

		# [num_worlds]
		fixed_s1_scores = s1_scores[:,qi,utt]

		# shape: [num_worlds]
		l1_posterior_unnormed = discrete_worlds_along_qud_prior + fixed_s1_scores
		# shape: [num_worlds]
		l1_posterior_normed = tf.exp(l1_posterior_unnormed - tf.reduce_logsumexp(l1_posterior_unnormed))

		# project the vectors lying along qud into the subspace
		# shape: [num_worlds]
		projected_worlds = tf.squeeze(projection_into_subspace_tf(tf.transpose(tf.squeeze(discrete_worlds_along_qud)),qud_matrix[qi]))

		#  shapes: scalar
		subspace_mean,subspace_variance = mean_and_variance_of_dist_array(probs=l1_posterior_normed,support=projected_worlds)

		print("subspace_mean",sess.run(subspace_mean))

		# [num dims-num_qud_dims,num_qud_dims]	
		orthogonal_dims = orthogonal_complement_tf(qud_matrix[qi])
		print("orthogonal_dims", orthogonal_dims)
		# add qud to list of basis vectors to obtain full basis
		orthogonal_basis = tf.concat([tf.transpose(orthogonal_dims),tf.transpose(qud_matrix[qi])],axis=0)
		#project the full space prior mean along each of the orthogonal vectors to the qud
		# shape: [num_dims-num_qud_dims,num_qud_dims]
		projected_orthogonal_means = tf.transpose(projection_into_subspace_tf(tf.expand_dims(listener_world,1),orthogonal_dims))
		print("projected_orthogonal_means", projected_orthogonal_means)

		# I use tf.diag to give the appropriate zeros, thus representing the basis in the NUM_DIMS dimensional space
		new_basis_means = tf.diag(tf.concat([projected_orthogonal_means[0],[subspace_mean]],axis=0))
		print("new_basis_means", new_basis_means)
		# sum all components
		new_basis_mean = tf.reduce_sum(new_basis_means,axis=0)

		# as before: just concatenate new variance with prior variance in each dimension
		new_basis_variance = tf.concat([tf.zeros([NUM_DIMS-NUM_QUD_DIMS])+inference_params.l1_sig1,[subspace_variance]],axis=0)

		print("mean and var",new_basis_mean,new_basis_variance)

		determinant = tf.matrix_determinant(tf.diag(new_basis_variance*2*pi))
		determinants.append(determinant)
		means.append(new_basis_mean)
		covariances.append(new_basis_variance)
		# define new normal in basis of qud and orthog dims
		# shape: [heatmap_size,num_dims]
		# shape: [heatmap_size]

		# print(sess.run([new_basis_mean,new_basis_variance]))
		# raise Exception


		if inference_params.heatmap:
	
			new_basis_normal=Normal(loc=new_basis_mean,scale=tf.sqrt(new_basis_variance))
			new_basis_support = tf.matmul(discrete_worlds,orthogonal_basis)
			heatmap_probs = tf.map_fn(lambda w: tf.reduce_sum(new_basis_normal.log_prob(w)),new_basis_support)
			heatmap = tf.reshape(heatmap_probs,(size*2+1,size*2+1))
			# 	# code ends here with return, now that the heatmap tensor has been obtained
			# 	return None,None		
			heatmaps.append(heatmap)




















		#CALCULATE FULL MEAN AND COVARIANCE FOR A SINGLE GAUSSIAN:

	

		# SHAPE of inference_params.qud_matrix[qi] : [NUM_DIMS, NUM_QUD_DIMS]

		# full_basis_qud_mean = tf.transpose(MEAN*inference_params.qud_matrix[qi])
			#multidim version
			# tf.matmul(MEAN,tf.transpose(inference_params.qud_matrix[qi]))
		# print("full_basis_qud_mean",full_basis_qud_mean)


		#could do this whole bit in numpy
		# THE ORTHOGONAL VECTORS TO CURRENT QUD Q
		# SHAPE: [NUM_DIMS,NUM_DIMS-NUM_QUD_DIMS]
		# current_qud = qud_matrix[qi] - tf.norm(qud_matrix[qi])
		# change for 2d
		# a = sess.run(tf.transpose(qud_matrix[qi]))
		# print(a,orthogonal_complement_np(a),orthogonal_complement_np(a.T))
		# orthogonal_dims = orthogonal_complement_tf(qud_matrix[qi])
		# print("shapes for proj",orthogonal_dims)

		# orthogonal_basis = tf.concat([orthogonal_dims,tf.transpose(qud_matrix[qi])],axis=0)
		# print("orthogonal_basis",orthogonal_basis)
		# print("qr check",np.identity(NUM_DIMS),sess.run(tf.matmul(tf.transpose(orthogonal_basis),orthogonal_basis)))
		
		# raise Exception
		# print("qr check",orthogonal_dims,qud_matrix[qi])
		# print("projection shapes 2",tf.expand_dims(listener_world,1),tf.transpose(orthogonal_dims))
		# projected_orthogonal_means = tf.transpose(projection_into_subspace_tf(tf.expand_dims(listener_world,1),orthogonal_dims))

		# print("full mean shapes",projected_orthogonal_means,orthogonal_dims)
		# PROJECT THE LISTENER WORLD (e.g. man in "man is shark") into each orthogonal_dim: 
		# SHAPE: [1, NUM_DIMS-NUM_QUD_DIMS, NUM_QUD_DIMS]
			# WOULD BE GOOD TO DOUBLE CHECK THIS!!
		# print("projected_orthogonal_means",projected_orthogonal_means)
		# ? WHAT IS THE OPERATION TO UNPROJECT THE PROJECTED MEANS
		# DO ELEMENTWISE MULTIPLICATION WITH ORTHONGAL DIMS

		#FIX: 

		# full_basis_orthogonal_means = tf.transpose(projected_orthogonal_means)*tf.transpose(orthogonal_dims)
			#multidim version
		# print("full_basis_orthogonal_means",full_basis_orthogonal_means)
		# raise Exception
		# concat_means = tf.concat([full_basis_orthogonal_means,full_basis_qud_mean],axis=0)
		# full_mean = tf.reduce_sum(concat_means,axis=0)
		# print("MEAN for qud"+str(qi),sess.run(full_mean))
		# means.append(full_mean)



		# WHAT ORDER SHOULD THESE BE IN?
		# covariance = tf.concat([tf.zeros([NUM_DIMS-NUM_QUD_DIMS])+inference_params.l1_sig1,VARIANCE],axis=0)
		
		# print("subspace variance for qud"+str(qi),sess.run(VARIANCE))
		# covariances.append(covariance)

		# two_pi_cov = 2*pi*covariance
		# print(covariance,"covariance")
		# determinant = tf.matrix_determinant(new_basis_variance*2*pi)
		# determinants.append(determinant)



	means = tf.stack(means)
	covariances = tf.stack(covariances)
	determinants = tf.stack(determinants)
	if inference_params.heatmap:
		stacked_worlds = tf.stack(heatmaps)

		# m = means[qi]

	# print(qud_combinations,sess.run(means),"MEANS")
	# raise Exception

	# a function which computes a number proportional to the integral of the area under the normal distribution
	#  corresponding to a particular qud
	# WAIT, not quite, why is w in there? i'm confused
	def qud_score(w,qi):



		det = determinants[qi]

		term_1 = tf.log(tf.sqrt(det))

		# print(w)
		# print(tf.expand_dims(w))

		term_2 = tf_s1(inference_params,s1_world=tf.expand_dims(w,0))[qi][utt]
		term_3 = tf.reduce_sum(full_space_prior.log_prob(w))

		# print(term_1,term_2,term_3, "terms")

		return term_1+term_2+term_3




	# print("test func", qud_score(means[0],0))

	qud_scores = tf.stack([qud_score(means[qi],qi) for qi in range(len(qud_combinations))])

	qud_distribution = qud_scores - tf.reduce_logsumexp(qud_scores,axis=0)

	print("stacked_worlds,qud_distribution",stacked_worlds,qud_distribution)
	inference_params.heatmap = sess.run(tf.reduce_sum(tf.exp(stacked_worlds)+tf.exp(tf.reshape(qud_distribution,(NUM_QUDS,1,1))),axis=0))

	# def posterior_likelihood(w,qi):

	# 	normal = Normal(loc=means[qi],scale=tf.sqrt(covariances[qi]))
	# 	orthogonal_dims = orthogonal_complement_tf(qud_matrix[qi])
	# 	# print("orthogonal_dims",orthogonal_dims)
	# 	orthogonal_basis = tf.concat([tf.transpose(orthogonal_dims),tf.transpose(qud_matrix[qi])],axis=0)
	# 	proj_w = projection_into_subspace_tf(tf.expand_dims(w,1),orthogonal_basis)
	# 	# print("proj_w",proj_w)
	# 	return tf.reduce_sum(normal.log_prob(proj_w))+qud_distribution[qi]

	# # print(sess.run(posterior_likelihood([0.0,1.0],0)),"posterior_likelihood")
	# # size,amount = inference_params.resolution
	# # discrete_worlds = np.asarray([[y*amount,x*amount] for (x,y) in itertools.product(range(-size,size),range(-size,size))],dtype=np.float32)
	
	# def world_sampler():
	# 	pass
		# qi = Categorical(logits=qud_distribution)
		# print(sess.run(qi))
		# print(sess.run(means[qi]))
		# normal = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=means[qi],covariance_matrix=covariances[qi])
		# return normal.sample()
		# pass
		# unit_gaussian = Normal(loc=0.0,scale=1.0)

		# # WRONG
		# # full_basis_qud_sample = tf.transpose(qworld.sample()*inference_params.qud_matrix[qi])
		# orthogonal_dims = orthogonal_complement_tf(qud_matrix[qi])
		# projected_orthogonal_means = projection_into_subspace_tf(tf.expand_dims(listener_world,1),orthogonal_dims)
		# samples = unit_gaussian.sample(NUM_DIMS-NUM_QUD_DIMS,1)
		# projected_orthogonal_samples = projected_orthogonal_means+(samples*inference_params.l1_sig1)
		# print("full mean shapes",projected_orthogonal_means,orthogonal_dims)
		# full_basis_orthogonal_samples = tf.transpose(projected_orthogonal_means)*tf.transpose(orthogonal_dims)
		# 	#multidim version
		# # print("full_basis_orthogonal_means",full_basis_orthogonal_means)
		# concat_samples = tf.concat([full_basis_orthogonal_samples,full_basis_qud_sample],axis=0)
		# full_samples = tf.reduce_sum(concat_samples,axis=0) 

		# return full_samples

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
	# world_samples = tf.map_fn(lambda x : posterior_likelihood(x,0),discrete_worlds)

	# world_samples=None

	# print(qud_distribution,np.exp(sess.run(qud_distribution)))

	# raise Exception
	results = list(zip(qud_combinations,sess.run(qud_distribution)))
	results = (sorted(results, key=lambda x: x[1], reverse=True))

	# print([(x,np.exp(y)) for (x,y) in results])

	# print(world_samples)
	# raise Exception
	return inference_params.heatmap,[(x,np.exp(y)) for (x,y) in results]
	# return sess.run(tf.reshape(world_samples,(size*2,size*2))),results



