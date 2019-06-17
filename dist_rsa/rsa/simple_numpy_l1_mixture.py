# uses tf but only for the s1

import time
import tqdm
import numpy as np
import scipy
import scipy.stats
from collections import Counter
import pickle
from scipy.stats import multivariate_normal
import tensorflow as tf
from dist_rsa.utils.helperfunctions import projection,tensor_projection,weights_to_dist,\
    normalize,as_a_matrix,tensor_projection_matrix,\
    double_tensor_projection_matrix,combine_quds, lookup,\
    s1_nonvect,double_tensor_projection_matrix_into_subspace,\
    orthogonal_complement_tf,projection_into_subspace_tf, projection_into_subspace_np, orthogonal_complement_np,\
    mean_and_variance_of_dist_array, mean_and_variance_of_dist_array_np
import itertools
from dist_rsa.rsa.tensorflow_s1 import tf_s1
from dist_rsa.rsa.tensorflow_s1_triple_vec import tf_s1 as tf_s1_triple_vec


def np_l1(inference_params):

	if inference_params.number_of_qud_dimensions!=1:
		raise Exception("Number of qud dimensions != 1")

	# NUM_QUDS: len(qud_combinations)
	# NUM_UTTS = number_of_utts
	NUM_DIMS = inference_params.subject_vector.shape[0]
	NUM_QUD_DIMS = inference_params.number_of_qud_dimensions

	#ALL THE SETUP STUFF
	sess = tf.Session()
	listener_world = inference_params.subject_vector
	number_of_qud_words = len(inference_params.quds)
	poss_utts = as_a_matrix(inference_params.possible_utterances,inference_params.vecs)
	number_of_utts = len(inference_params.possible_utterances)
	poss_utt_freqs = np.log(np.array([lookup(inference_params.poss_utt_frequencies,x,'NOUN') for x in inference_params.possible_utterances]))
	weighted_utt_frequency_array = poss_utt_freqs-scipy.misc.logsumexp(poss_utt_freqs)
	qud_weight = inference_params.qud_weight

	if not inference_params.qud_frequencies:
		qud_frequencies = inference_params.poss_utt_frequencies
	#order is wrong: not a problem when freqs are always one, but otherwise v bad
	qud_freqs = -np.log(np.array([lookup(inference_params.qud_frequencies,x,'ADJ') for x in inference_params.quds]))
	weighted_qud_frequency_array = np.expand_dims(qud_freqs-scipy.misc.logsumexp(qud_freqs),1)
	qud_combinations = combine_quds(inference_params.quds,inference_params.number_of_qud_dimensions)
	inference_params.qud_combinations = qud_combinations
	NUM_QUDS = len(qud_combinations)
	print("qud_combinations",NUM_QUDS,qud_combinations)
	print("quds",len(inference_params.quds))
	# SHAPE: [NUM_QUDS, NUM_DIMS, NUM_QUD_DIMS]
	qud_matrix = (np.asarray([np.asarray([inference_params.vecs[word]/np.linalg.norm(inference_params.vecs[word]) for word in words]).T for words in qud_combinations]))
	inference_params.weighted_utt_frequency_array=weighted_utt_frequency_array
	inference_params.poss_utts=poss_utts
	inference_params.listener_world=listener_world 
	u=inference_params.predicate
	utt = inference_params.possible_utterances.index(u)
	pi = np.pi
	size,amount = inference_params.resolution.size, inference_params.resolution.amount
	print("SIZE,AMOUNT",size,amount)
	determinants=[]
	covariances=[]
	means=[]
	heatmaps = []
	orthogonal_basis_for_each_qud = []

	if inference_params.heatmap:
		discrete_worlds = np.asarray([[x*amount,y*amount] for (x,y) in itertools.product(range(-size,size+1),range(-size,size+1))])

	# CALCULATE MEAN AND SIGMA OF p(w|q=qi,u=u) in the projected space
	# CODE IS ONLY FOR 1D QUDS
	full_space_prior = multivariate_normal(mean=np.squeeze(listener_world), cov=[inference_params.l1_sig1] * inference_params.vec_length)
	tic = time.time()
	for qi in tqdm.tqdm(range(len(qud_combinations))):

		print("CURRENT QUD:", qud_combinations[qi])

		projected_listener_world = projection_into_subspace_np(np.expand_dims(listener_world,1),qud_matrix[qi])			

		approximate_mean = 0.0

		optimize = True
		if optimize:

			# the discrete support: tf.stack turns list of vectors (each a point lying along qud) into tensor where each row is vector
			# shape: [num_worlds,num_qud_dims,num_dims] (n.b. num_worlds = size*2+1)
			discrete_worlds_along_qud = np.asarray([np.transpose(qud_matrix[qi])*((amount*x)+approximate_mean) for x in range(-size,size+1)])
			# SHOULDN'T I TAKE LOG PDF??
			discrete_worlds_along_qud_prior = np.asarray([np.log(full_space_prior.pdf(w)) for w in discrete_worlds_along_qud])
			inference_params.qud_matrix = tf.expand_dims(qud_matrix[qi],0)
			t1 = time.time()
			s1_scores = sess.run(tf_s1_triple_vec(inference_params,s1_world=np.squeeze(discrete_worlds_along_qud),NUMPY=True))
			fixed_s1_scores = s1_scores[:,0,utt]
			l1_posterior_unnormed = discrete_worlds_along_qud_prior + fixed_s1_scores
			l1_posterior_normed = l1_posterior_unnormed - scipy.misc.logsumexp(l1_posterior_unnormed)
			approximate_mean = (np.arange(-size,size+1)*amount)[np.argsort(-l1_posterior_normed)[0]]
		
		# see how far you need to go out
		# print(sess.run(f_w))
		# span = tf.Variable(10.0)
		# init = tf.global_variables_initializer()
		# sess.run(init)
		# left_w = tf.transpose(inference_params.qud_matrix[qi]*(approximate_mean-span))
		# right_w = tf.transpose(inference_params.qud_matrix[qi]*(approximate_mean+span))
		# left_point = tf.reduce_sum(full_space_prior.log_prob(left_w))+tf_s1(inference_params,s1_world=w)[qi,utt]
		# right_point =tf.reduce_sum(full_space_prior.log_prob(right_w))+tf_s1(inference_params,s1_world=w)[qi,utt]
		
		# left_cond = tf.exp(f_w)/tf.exp(left_point)>1000
		# right_cond = tf.exp(f_w)/tf.exp(right_point)>1000
		# cond = tf.logical_not(tf.logical_and(left_cond, right_cond))
		# # print(sess.run([f_w,right_point,tf.exp(f_w)/tf.exp(right_point),tf.exp(f_w)/tf.exp(right_point)>1000]))
		# print(sess.run(cond))
		# span_op = span.assign(span+1.0)
		# sess.run(span_op)
		# print(sess.run(span))


		discrete_worlds_along_qud = np.asarray([np.transpose(qud_matrix[qi])*((amount*x)+approximate_mean) for x in range(-size,size+1)])
		discrete_worlds_along_qud_prior = np.asarray([np.log(full_space_prior.pdf(w)) for w in discrete_worlds_along_qud])
		# shape: [num_worlds,num_quds,num_utts] : but note, num_quds=1 here: we are fixing a qud
		inference_params.qud_matrix = tf.expand_dims(qud_matrix[qi],0)

		# s1_scores =  tf.map_fn(lambda w: tf_s1(inference_params,s1_world=w),discrete_worlds_along_qud)
		t1 = time.time()
		s1_scores = sess.run(tf_s1_triple_vec(inference_params,s1_world=np.squeeze(discrete_worlds_along_qud),NUMPY=True))
		print("s1 time",time.time()-t1)
		# [num_worlds]
		fixed_s1_scores = s1_scores[:,0,utt]


		# print(discrete_worlds_along_qud_prior)
		# print("s1_scores",fixed_s1_scores)
		# raise Exception

		# shape: [num_worlds]
		# print(discrete_worlds_along_qud_prior)
		# print(fixed_s1_scores)
		l1_posterior_unnormed = discrete_worlds_along_qud_prior + fixed_s1_scores

		# print("OPTIMIZATION")
		# print("discrete_worlds_along_qud", discrete_worlds_along_qud)
		# print(np.gradient(l1_posterior_unnormed),discrete_worlds_along_qud[np.argmax(-np.abs(np.gradient(l1_posterior_unnormed)))])
		# raise Exception

		# shape: [num_worlds]
		l1_posterior_normed = l1_posterior_unnormed - scipy.misc.logsumexp(l1_posterior_unnormed)
		# print(np.arange(-size,size+1)*amount)
		# print((np.arange(-size,size+1)*amount)[np.argsort(-l1_posterior_normed)[0]])
		# raise Exception
		# project the vectors lying along qud into the subspace
		# shape: [num_worlds]
		projected_worlds = np.squeeze(projection_into_subspace_np(np.transpose(np.squeeze(discrete_worlds_along_qud)),qud_matrix[qi]))
		#  shapes: scalar
		subspace_mean,subspace_variance = mean_and_variance_of_dist_array_np(probs=np.exp(l1_posterior_normed),support=projected_worlds)
		

		# [num dims-num_qud_dims,num_qud_dims]	
		orthogonal_dims = orthogonal_complement_np(qud_matrix[qi])
		# add qud to list of basis vectors to obtain full basis: each row is a basis vector
		orthogonal_basis = np.concatenate([np.transpose(orthogonal_dims),np.transpose(qud_matrix[qi])],axis=0)
		#project the full space prior mean along each of the orthogonal vectors to the qud
		# shape: [num_dims-num_qud_dims,num_qud_dims]

		# NOTE THAT THIS RELIES ON THE FACT THAT THE MATRIX I'M PROJECTING ONTO IS OF ORTHOGONAL VECTORS
		projected_orthogonal_means = np.transpose(projection_into_subspace_np(np.expand_dims(listener_world,1),orthogonal_dims))
		
		# I use tf.diag to give the appropriate zeros, thus representing the basis in the NUM_DIMS dimensional space
		new_basis_means = np.diag(np.concatenate([projected_orthogonal_means[0],[subspace_mean]],axis=0))
		# sum all components
		new_basis_mean = np.sum(new_basis_means,axis=0)
		old_basis_mean = np.matmul(np.transpose(orthogonal_basis),np.expand_dims(new_basis_mean,1))

		# concatenate new variance with prior variance in each dimension
		new_basis_variance = np.concatenate([np.zeros([NUM_DIMS-NUM_QUD_DIMS])+inference_params.l1_sig1,[subspace_variance]],axis=0)

		determinant = np.sum(np.log(new_basis_variance*2*pi))
		# print("new basis variance",new_basis_variance)
		# print("determinant",determinant)
		determinants.append(determinant)
		means.append(np.squeeze(old_basis_mean))
		covariances.append(new_basis_variance)
		orthogonal_basis_for_each_qud.append(orthogonal_basis)


		if inference_params.heatmap:
			raise Exception("not yet implemented")
	
			# define new normal in basis of qud and orthog dims
			new_basis_normal=multivariate_normal(loc=new_basis_mean,scale=new_basis_variance)
			new_basis_support = tf.matmul(discrete_worlds,orthogonal_basis)
			heatmap_probs = tf.map_fn(lambda w: tf.reduce_sum(new_basis_normal.log_prob(w)),new_basis_support)
			heatmap = tf.transpose(tf.reshape(heatmap_probs,(size*2+1,size*2+1)) - tf.reduce_logsumexp(heatmap_probs))
			heatmaps.append(heatmap)
		else: heatmaps = None


	means = np.asarray(means)
	# print("means",means)
	# raise Exception
	covariances = np.asarray(covariances)
	determinants = np.asarray(determinants)
	orthogonal_basis_for_each_qud = np.asarray(orthogonal_basis_for_each_qud)
	toc = time.time()
	print("time:",toc-tic)

	# a function which computes a number proportional to the integral of the area under the normal distribution
	# corresponding to a particular qud
	def qud_score(qi):

		det = determinants[qi]
		term_1 = det/2
		inference_params.qud_matrix = tf.expand_dims(qud_matrix[qi],0)
		term_2 = sess.run(tf_s1(inference_params,s1_world=np.expand_dims(means[qi],0),NUMPY=True)[0][utt])
		# print("check 2")
		term_3 = np.log(full_space_prior.pdf(means[qi]))
		# print("check 3")
		# print(term_1)
		# print(term_2)
		# print(term_3)
		# raise Exception
		# print(term_1)

		return term_1+term_2+term_3
	# print("check 1")
	qud_scores = np.asarray([qud_score(qi) for qi in range(len(qud_combinations))])
	# print("check 4")
	# print(qud_scores[0])
	# print("scores",qud_scores)
	print(scipy.misc.logsumexp(qud_scores,axis=0))
	qud_distribution = qud_scores - scipy.misc.logsumexp(qud_scores,axis=0)
	# print("check 5")
	qud_distribution_np = np.exp(qud_distribution)
	inference_params.ordered_quds = list(list(zip(sorted(list(zip(qud_combinations, qud_distribution_np)), key=lambda x: x[-1], reverse=True)))[0])
	# print("check 6")
	inference_params.qud_marginals=qud_distribution_np
	tac = time.time()
	print("time:",tac-toc)



	if inference_params.heatmap:
		raise Exception("not yet implemented")
		stacked_worlds = tf.stack(heatmaps)
		inference_params.heatmaps=sess.run(tf.exp(stacked_worlds))
		worlds =   sess.run(tf.reduce_sum(tf.exp(stacked_worlds+tf.reshape(qud_distribution,(NUM_QUDS,1,1))),axis=0))
		mean,_ = sess.run(mean_and_variance_of_dist_array(probs=np.reshape(worlds,newshape=((size*2+1)*(size*2+1),1)),support=discrete_worlds))
		inference_params.worlds=worlds
		inference_params.heatmap_mean=mean
		print("HEATMAP MEAN",mean)
	else: worlds = None
	conditionals=["n/a"]*NUM_QUDS






	def calculate_mean_and_variance_in_subspace(subspace_vector,qud_index):
		# takes: 
			# a representation of the subspace in the original basis
			# index of a qud
		# returns:
			# a mean and variance in the basis of the subspace
		qud_vector = qud_matrix_np[qud_index]
		line_in_orthogonal_basis = np.dot(orthogonal_basis_for_each_qud_np[qud_index],subspace_vector)
		covariance = np.diag(covariances_np[qud_index])
		variance = np.dot(np.dot(np.transpose(line_in_orthogonal_basis),covariance),line_in_orthogonal_basis)
		projected_mean = projection_into_subspace_np(np.expand_dims(means_np[qud_index],1),subspace_vector)
		projected_prior_mean = projection_into_subspace_np(np.expand_dims(inference_params.subject_vector,1),subspace_vector)
		return np.squeeze(projected_mean),np.squeeze(projected_prior_mean),np.squeeze(variance)


	if inference_params.calculate_projected_marginal_world_posterior:
		means_np,orthogonal_basis_for_each_qud_np,qud_matrix_np,covariances_np = means,orthogonal_basis_for_each_qud,qud_matrix,covariances
		inference_params.means = means_np
		subspace_means = np.zeros((NUM_QUDS,NUM_QUDS))
		subspace_prior_means = np.zeros((NUM_QUDS,NUM_QUDS))
		subspace_variances = np.zeros((NUM_QUDS,NUM_QUDS))

		for line in range(NUM_QUDS):

			for qud_index in range(NUM_QUDS):

				subspace_mean,subspace_prior_mean,subspace_variance = calculate_mean_and_variance_in_subspace(
					subspace_vector=qud_matrix_np[line],
					qud_index=qud_index,
					)

				subspace_means[line,qud_index]=subspace_mean
				subspace_prior_means[line,qud_index]=subspace_prior_mean
				subspace_variances[line,qud_index]=subspace_variance

		inference_params.marginal_means = np.sum(subspace_means.T*np.expand_dims(qud_distribution_np,1),axis=0)
		inference_params.subspace_means=subspace_means
		inference_params.subspace_prior_means=subspace_prior_means
		inference_params.subspace_variances=subspace_variances



	# tuc = time.time()
	# print("time:",tuc-tac)
	results = list(zip(qud_combinations, qud_distribution_np))
	results = (sorted(results, key=lambda x: x[-1], reverse=True))
	print(results)
	sess.close()
	# return results



