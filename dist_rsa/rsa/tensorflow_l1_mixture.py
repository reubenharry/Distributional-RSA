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
    mean_and_variance_of_dist_array,mean_and_variance_of_dist_array_np
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
	# listener_world = tf.zeros((300))
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
	weighted_qud_frequency_array = tf.cast(tf.expand_dims(qud_freqs-scipy.misc.logsumexp(qud_freqs),1),dtype=tf.float32)
	qud_combinations = combine_quds(inference_params.quds,inference_params.number_of_qud_dimensions)
	NUM_QUDS = len(qud_combinations)
	print("qud_combinations",NUM_QUDS,qud_combinations)
	print("quds",len(inference_params.quds))
	# SHAPE: [NUM_QUDS, NUM_DIMS, NUM_QUD_DIMS]
	qud_matrix = tf.cast((np.asarray([np.asarray([inference_params.vecs[word]/np.linalg.norm(inference_params.vecs[word]) for word in words]).T for words in qud_combinations])),dtype=tf.float32)
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
	print("QUD MATRIX SHAPE",qud_matrix)
	# projected_listener_worlds = projection_into_subspace_tf(tf.expand_dims(listener_world,1),tf.transpose(qud_matrix[:,:,0]))
	pi = tf.constant(np.pi)
	size,amount = inference_params.resolution.size, inference_params.resolution.amount
	print("SIZE,AMOUNT",size,amount)
	determinants=[]
	covariances=[]
	means=[]
	heatmaps = []
	std = tf.sqrt(inference_params.l1_sig1)

	if inference_params.heatmap:
		discrete_worlds = tf.constant(np.asarray([[y*amount,x*amount] for (x,y) in itertools.product(range(-size,size+1),range(-size,size+1))],dtype=np.float32))

		print("discrete_worlds",discrete_worlds)

	# CALCULATE MEAN AND SIGMA OF p(w|q=qi,u=u) in the projected space
	#ALL THIS CODE IS WRONG FOR MULTIDIMENSIONAL QUDS
	full_space_prior = Normal(loc=tf.squeeze(listener_world), scale=[std] * inference_params.vec_length)
	for qi in range(len(qud_combinations)):
		
		print(qud_combinations[qi],"CURRENT QUD")
		# projected_listener_world = projected_listener_worlds[qi]	
		projected_listener_world = projection_into_subspace_tf(tf.expand_dims(listener_world,1),qud_matrix[qi])			
		# the discrete support: tf.stack turns list of vectors (each a point lying along qud) into tensor where each row is vector
		# shape: [num_worlds,num_qud_dims,num_dims] (n.b. num_worlds = size*2+1)


		# w = tf.transpose(inference_params.qud_matrix[qi]*amount*10)
		

		# def f(x):
		# 	return tf.exp(tf_s1(inference_params,s1_world=x)[qi,utt])



				# convex optimization, nbd
		approximate_mean = tf.Variable(0.0)
		w = tf.transpose(inference_params.qud_matrix[qi]*approximate_mean)
		init = tf.global_variables_initializer()
		sess.run(init)
		f_w = tf.reduce_sum(full_space_prior.log_prob(w))+tf_s1(inference_params,s1_world=w)[qi,utt]
		# print("fw",sess.run(f_w))
		grad = tf.gradients(f_w,approximate_mean)[0]
		# print("original grad",sess.run(grad))
		optimize_op = approximate_mean.assign(approximate_mean + 0.01 * grad)

		for i in range(1000):
			sess.run(optimize_op)
			# print(sess.run(x))

		print("grad and mean",sess.run([grad,approximate_mean]))

		discrete_worlds_along_qud = tf.stack([tf.transpose(inference_params.qud_matrix[qi])*((amount*x)+approximate_mean) for x in range(-size,size+1)])
		# shape: [num_worlds]
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
		


		# raise Exception
		# if sess.run

		discrete_worlds_along_qud_prior = tf.map_fn(lambda w: tf.reduce_sum(full_space_prior.log_prob(w)),discrete_worlds_along_qud)
		# shape: [num_worlds,num_quds,num_utts]
		s1_scores =  tf.map_fn(lambda w: tf_s1(inference_params,s1_world=w),discrete_worlds_along_qud)
		# [num_worlds]
		fixed_s1_scores = s1_scores[:,qi,utt]

		# shape: [num_worlds]
		l1_posterior_unnormed = discrete_worlds_along_qud_prior + fixed_s1_scores
		# print("l1_post",sess.run((l1_posterior_unnormed)))
		# print("args", np.argsort(sess.run((l1_posterior_unnormed))))
		# shape: [num_worlds]
		l1_posterior_normed = l1_posterior_unnormed - tf.reduce_logsumexp(l1_posterior_unnormed)
		# project the vectors lying along qud into the subspace
		# shape: [num_worlds]
		projected_worlds = tf.squeeze(projection_into_subspace_tf(tf.transpose(tf.squeeze(discrete_worlds_along_qud)),qud_matrix[qi]))
		#  shapes: scalar
		subspace_mean,subspace_variance = mean_and_variance_of_dist_array(probs=tf.exp(l1_posterior_normed),support=projected_worlds)
		
		print(sess.run([subspace_mean]),"subspace mean")

		# [num dims-num_qud_dims,num_qud_dims]	
		orthogonal_dims = orthogonal_complement_tf(qud_matrix[qi])
		# add qud to list of basis vectors to obtain full basis: each row is a basis vector
		orthogonal_basis = tf.concat([tf.transpose(orthogonal_dims),tf.transpose(qud_matrix[qi])],axis=0)
		#project the full space prior mean along each of the orthogonal vectors to the qud
		# shape: [num_dims-num_qud_dims,num_qud_dims]
		

		# projected_orthogonal_means_list = []
		# # print("orth dims shape:",orthogonal_dims)
		# for dim_iter in range(orthogonal_dims.get_shape()[1]):
		# 	# print("SHAPE1",tf.expand_dims(listener_world,1))
		# 	print("SHAPE2", tf.expand_dims(tf.transpose(orthogonal_dims)[dim_iter],1) )
		# 	# print("KEY SPAE", projection_into_subspace_tf(tf.expand_dims(listener_world,1),tf.expand_dims(tf.transpose(orthogonal_dims)[dim_iter],1)))
		# 	projected_orthogonal_means_list.append(projection_into_subspace_tf(tf.expand_dims(listener_world,1),tf.expand_dims(tf.transpose(orthogonal_dims)[dim_iter],1))[0])
		# projected_orthogonal_means_alt = tf.transpose(tf.stack(projected_orthogonal_means_list))
		
		# NOTE THAT THIS RELIES ON THE FACT THAT THE MATRIX I'M PROJECTING ONTO IS OF ORTHOGONAL VECTORS
		projected_orthogonal_means = tf.transpose(projection_into_subspace_tf(tf.expand_dims(listener_world,1),orthogonal_dims))
		# print(projected_orthogonal_means)
		# print("PROJ ORTH MEANS",sess.run(projected_orthogonal_means))
		# print("key shape",orthogonal_dims)
		# print("PROJ ORTH MEANS ALT",sess.run(projected_orthogonal_means_alt))
		# raise Exception
		
		# I use tf.diag to give the appropriate zeros, thus representing the basis in the NUM_DIMS dimensional space
		new_basis_means = tf.diag(tf.concat([projected_orthogonal_means[0],[subspace_mean]],axis=0))
		# sum all components
		new_basis_mean = tf.reduce_sum(new_basis_means,axis=0)
		# print("new_basis_mean", new_basis_mean)
		# print("trans orth basis",tf.transpose(orthogonal_basis))
		old_basis_mean = tf.einsum('n,nm->m', new_basis_mean, tf.transpose(orthogonal_basis))

		print("shape",old_basis_mean)
		print("projected", sess.run(projection_into_subspace_tf(tf.expand_dims(old_basis_mean,1),qud_matrix[0])))

		# concatenate new variance with prior variance in each dimension
		new_basis_variance = tf.concat([tf.zeros([NUM_DIMS-NUM_QUD_DIMS])+inference_params.l1_sig1,[subspace_variance]],axis=0)


		# determinant = tf.matrix_determinant(tf.diag(new_basis_variance*2*pi))
		# determinant = tf.reduce_prod(new_basis_variance*2*pi)
		#NB I put the variance into log space before computing determinant
		determinant = tf.reduce_sum(tf.log(new_basis_variance*2*pi))
		determinants.append(determinant)
		# print("old basis mean",sess.run(old_basis_mean),old_basis_mean)
		means.append(old_basis_mean)
		covariances.append(new_basis_variance)


		if inference_params.heatmap:
	
			# define new normal in basis of qud and orthog dims
			new_basis_normal=Normal(loc=new_basis_mean,scale=tf.sqrt(new_basis_variance))
			new_basis_support = tf.matmul(discrete_worlds,orthogonal_basis)
			heatmap_probs = tf.map_fn(lambda w: tf.reduce_sum(new_basis_normal.log_prob(w)),new_basis_support)
			heatmap = tf.reshape(heatmap_probs,(size*2+1,size*2+1))
			# 	# code ends here with return, now that the heatmap tensor has been obtained
			# 	return None,None		
			heatmaps.append(heatmap)


	means = tf.stack(means)
	covariances = tf.stack(covariances)
	determinants = tf.stack(determinants)

	# a function which computes a number proportional to the integral of the area under the normal distribution
	#  corresponding to a particular qud
	def qud_score(qi):

		det = determinants[qi]
		# print("det",sess.run(det))
		term_1 = det/2
		term_2 = tf_s1(inference_params,s1_world=tf.expand_dims(means[qi],0))[qi][utt]
		term_3 = tf.reduce_sum(full_space_prior.log_prob(means[qi]))

		# print("QUD",qud_combinations[qi])
		# print("DETERMINANT",sess.run(determinants[qi]))
		# print("MEAN",sess.run(means[qi]))
		# print("QUD SCORE TERMS",sess.run([term_1,term_2,term_3]))
		# print("QUD SCORE",sess.run(term_1+term_2+term_3))
		# print("UTT LIKELiHOOD",sess.run(tf_s1(inference_params,s1_world=tf.expand_dims(tf.constant(inference_params.vecs[u],dtype=tf.float32),0))))

		# gaussian(mean=means[qi],variance=)

		return term_1+term_2+term_3

	qud_scores = tf.stack([qud_score(qi) for qi in range(len(qud_combinations))])
	# print(sess.run(qud_scores),"scores")
	qud_distribution = qud_scores - tf.reduce_logsumexp(qud_scores,axis=0)

	if inference_params.heatmap:
		stacked_worlds = tf.stack(heatmaps)
		# CONSULT LEON
		worlds = tf.reduce_sum(tf.exp(stacked_worlds+tf.reshape(qud_distribution,(NUM_QUDS,1,1))),axis=0)

		# worlds = sess.run(tf.reduce_sum(tf.exp(stacked_worlds)+tf.exp(tf.reshape(qud_distribution,(NUM_QUDS,1,1))),axis=0))
	else:
		worlds = tf.constant(1)
	def world_sampler():
		pass

	results = list(zip(qud_combinations,sess.run(qud_distribution)))
	results = (sorted(results, key=lambda x: x[1], reverse=True))

	# print("means",sess.run(means))

	marginal_mean = tf.reduce_sum(means*tf.expand_dims(tf.exp(qud_distribution),1),axis=0)

	mus = tf.divide(tf.add(inference_params.listener_world/inference_params.sigma1,inference_params.poss_utts/inference_params.sigma2),inference_params.inverse_sd)

	# from dist_rsa.utils.load_data import load_vecs
	# vec_size,vec_kind = 300,'glove.6B.'
	# vecs = load_vecs(mean=True,pca=False,vec_length=vec_size,vec_type=vec_kind) 

	for i in range(NUM_QUDS):




		# print(sess.run([inference_params.mus[utt],inference_params.projected_mus[i][utt],projection_into_subspace_tf(tf.expand_dims(inference_params.mus[i],1),qud_matrix[i])]))
		# print("PROJECTION")
		# print(qud_combinations[i],sess.run([means[i]]))
		print("QUD",qud_combinations[i])
		print("PRIOR",sess.run([projection_into_subspace_tf(tf.expand_dims(listener_world,1),qud_matrix[i])]))
		print("MUS",sess.run([projection_into_subspace_tf(tf.expand_dims(mus[utt],1),qud_matrix[i])]))

		# print("man on qud",sess.run(projection_into_subspace_tf(tf.expand_dims(vecs['man'],1),tf.expand_dims(vecs[qud_combinations[i][0]],1))))
		# print("man on qud",sess.run(projection_into_subspace_tf(tf.expand_dims(tf.cast(vecs['man'],dtype=tf.float32),1),qud_matrix[i])))
		# print("difference of differences",scipy.spatial.distance.cosine())

		# print("MUS",sess.run([projection_into_subspace_tf(tf.expand_dims(inference_params.mus[i],1),qud_matrix[i])]))
		print("CONDITIONAL MEAN")
		print(sess.run(projection_into_subspace_tf(tf.expand_dims(means[i],1),qud_matrix[i])))
		print("MARGINAL MEAN")
		print(sess.run(projection_into_subspace_tf(tf.expand_dims(marginal_mean,1),qud_matrix[i])))

	return sess.run(means), sess.run(worlds),[(x,np.exp(y)) for (x,y) in results]



