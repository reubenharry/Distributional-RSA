# if you're not comfortable with einstein summation, this vectorized implementation of the s1 may appear inscrutable. 
# For the semantics of the s1, try looking at the numpy version instead (numpy_rsa.py)


def tf_s1(inference_params,s1_world,world_movement=False,debug=False):

	# print("shape triple_vec",s1_world)

	from dist_rsa.utils.helperfunctions import projection,tensor_projection,weights_to_dist,\
		normalize,as_a_matrix,tensor_projection_matrix,\
		double_tensor_projection_matrix,combine_quds, lookup, s1_nonvect
	import tensorflow as tf
	import edward as ed
	import numpy as np
	import scipy
	from dist_rsa.utils.vectorized_subtraction import fast_vectorized_subtraction

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
	mus = tf.divide(tf.add(inference_params.listener_world/inference_params.sigma1, 
		inference_params.poss_utts/inference_params.sigma2),inference_params.inverse_sd)

	s1_world = tf.transpose(s1_world)
	mus = tf.transpose(mus) #mus now contains a listener interpretation in each column
	projected_mus = tf.transpose(tf.einsum('aij,jk->aik',qud_projection_matrix,mus),perm=[0,2,1])
	projected_world = tf.transpose(tf.einsum('aij,jk->aik',qud_projection_matrix,s1_world),perm=[0,2,1])
	if debug:
		# pass
		print("tf projected_world tv", ed.get_session().run(projected_world),projected_world)
		print("PROJECTED MUS tv",ed.get_session().run(projected_mus),projected_mus)


	# new_projected_world = tf.transpose(tf.reshape(tf.tile(projected_world,[1,tf.shape(projected_mus)[1],1]), shape=[tf.shape(projected_world)[0],tf.shape(projected_mus)[1],tf.shape(projected_world)[1],tf.shape(projected_world)[2]]),perm=[1,0,2,3])
	# new_projected_mus = tf.expand_dims(projected_mus,0)
	# tf.reshape(tf.tile(projected_mus,[1,tf.shape(projected_world)[1],1]), shape=[tf.shape(projected_world)[1],tf.shape(projected_mus)[0],tf.shape(projected_mus)[1],tf.shape(projected_mus)[2]])

	# print("new projected world",ed.get_session().run(tf.shape(new_projected_world)),"new projected mus",ed.get_session().run(tf.shape(new_projected_mus)))
	mus_world_diff = tf.transpose(fast_vectorized_subtraction(projected_mus,projected_world),perm=[0,1,3,2])
	# if debug:
	# 	print("mus_world_diff tv",mus_world_diff,ed.get_session().run(mus_world_diff))

	rescaled_mu_diffs = tf.einsum('ij,bajk->baik',tf.sqrt(inference_params.inverse_sigma),mus_world_diff) #note that this is only correct if sigma is diagonal
	squared_diffs = tf.square(rescaled_mu_diffs)
	log_likelihoods = tf.multiply(-0.5,tf.reduce_sum(squared_diffs,axis=2))
	
	utterance_scores = tf.multiply(inference_params.rationality, tf.add(log_likelihoods,tf.multiply(inference_params.freq_weight,inference_params.weighted_utt_frequency_array)))
	if debug:
		# pass
		# print("tf new_projected_mus tv", ed.get_session().run(new_projected_mus))
		# print("tf new_projected_world tv", ed.get_session().run(new_projected_world))
		print("tf log_likelihoods",ed.get_session().run(log_likelihoods))

		# print("tf scores",ed.get_session().run(utterance_scores))
	#this is where the rationality parameter is included
	norm = tf.expand_dims(tf.reduce_logsumexp(utterance_scores,axis=-1),-1)

	# print(utterance_scores,norm,"utt scores, norm")

	# new_projected_world = tf.reshape(tf.tile(projected_world,[1,tf.shape(projected_mus)[1],1]), shape=[tf.shape(projected_mus)[1],tf.shape(projected_world)[0],tf.shape(projected_world)[1],tf.shape(projected_world)[2]])
	# new_projected_mus = tf.reshape(tf.tile(projected_mus,[1,tf.shape(projected_world)[1],1]), shape=[tf.shape(projected_world)[1],tf.shape(projected_mus)[0],tf.shape(projected_mus)[1],tf.shape(projected_mus)[2]])

	out = tf.subtract(utterance_scores, norm)
	# print("out pre transpose",out)
	# print(out,"shapey")
	return tf.transpose(out,[1,0,2])