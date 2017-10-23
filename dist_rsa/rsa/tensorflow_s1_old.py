# if you're not comfortable with einstein summation, this vectorized implementation of the s1 may appear inscrutable. 
# For the semantics of the s1, try looking at the numpy version instead (numpy_rsa.py)


def tf_s1_old(inference_params,s1_world,world_movement=False,debug=False):

	from dist_rsa.utils.helperfunctions import projection,tensor_projection,weights_to_dist,\
		normalize,as_a_matrix,tensor_projection_matrix,\
		double_tensor_projection_matrix,combine_quds, lookup, s1_nonvect
	import tensorflow as tf
	import edward as ed
	import numpy as np
	import scipy

	def l0_sigma(sig1=1.0,sig2=1.0):
		sigma1 = sig1 ** 2
		sigma2 = sig2 ** 2
		inverse_sd = (1/sigma1) + (1/sigma2)
		sigma = tf.diag([tf.pow(inverse_sd,-1)] * inference_params.vec_length)
		inverse_sigma = tf.matrix_inverse(sigma)
		# print("sigma info",sigma1,sigma2,inverse_sd,sigma)
		return sigma1,sigma2,inverse_sd,sigma,inverse_sigma

	inference_params.sigma1,inference_params.sigma2,inference_params.inverse_sd,inference_params.sigma,inference_params.inverse_sigma = l0_sigma(sig1=inference_params.sig1,sig2=inference_params.sig2)


	# print("tf qud mat", inference_params.qud_matrix, ed.get_session().run(inference_params.qud_matrix))
	qud_projection_matrix = double_tensor_projection_matrix(inference_params.qud_matrix)
	mus = tf.divide(tf.add(inference_params.listener_world/inference_params.sigma1, inference_params.poss_utts/inference_params.sigma2),  inference_params.inverse_sd)
	mus = tf.transpose(mus) #mus now contains a listener interpretation in each column
	projected_mus = tf.transpose(tf.einsum('aij,jk->aik',qud_projection_matrix,mus),perm=[0,2,1])
	s1_world = tf.transpose(s1_world)
	projected_world = tf.transpose(tf.einsum('aij,jk->aik',qud_projection_matrix,s1_world),perm=[0,2,1])
	if debug:
		print("tf projected_world dv", ed.get_session().run(projected_world),projected_world)
		print("PROJECTED MUS dv",ed.get_session().run(projected_mus),projected_mus)

	mus_world_diff = tf.transpose(tf.subtract(projected_mus,projected_world),perm=[0,2,1])
	rescaled_mu_diffs = tf.einsum('ij,ajk->aik',tf.sqrt(inference_params.inverse_sigma),mus_world_diff) #note that this is only correct if sigma is diagonal
	squared_diffs = tf.square(rescaled_mu_diffs)
	log_likelihoods = tf.multiply(-0.5,tf.reduce_sum(squared_diffs,axis=1))
	if debug:
		# pass
		print("tf log_likelihoods",ed.get_session().run(log_likelihoods))


	utterance_scores = tf.add(log_likelihoods,tf.multiply(inference_params.freq_weight,inference_params.weighted_utt_frequency_array))
	norm = tf.expand_dims(tf.reduce_logsumexp(utterance_scores,axis=1),1)
	out = tf.subtract(utterance_scores, norm)
	return out