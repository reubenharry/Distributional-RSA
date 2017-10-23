# if you're not comfortable with einstein summation, this vectorized implementation of the s1 may appear inscrutable. 
# For the semantics of the s1, try looking at the numpy version instead (numpy_rsa.py)


def tf_s1_trivial(inference_params,s1_world,world_movement=False,debug=False):


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

	qud_projection_matrix = double_tensor_projection_matrix(inference_params.qud_matrix)
	mus = tf.divide(tf.add(inference_params.listener_world/inference_params.sigma1, 
		inference_params.poss_utts/inference_params.sigma2),inference_params.inverse_sd)

	if world_movement:
		print("LISTENER MUS SHAPE",mus.shape)
		words_with_distance_to_prior = np.asarray(list(map(lambda x: scipy.spatial.distance.cosine(inference_params.vecs[x],ed.get_session().run(inference_params.listener_world)),inference_params.quds)))
		words_with_distance_to_posterior = np.asarray(list(map((lambda y:   list(map((lambda x: scipy.spatial.distance.cosine(inference_params.vecs[x],y)),inference_params.quds))),ed.get_session().run(mus))))
		word_movement = words_with_distance_to_posterior-words_with_distance_to_prior
		mean_word_movement = np.squeeze(np.mean(word_movement,axis=0))
		out = sorted(list(zip(inference_params.quds,mean_word_movement)),key=lambda x:x[1],reverse=True)
		print("L0 world movement:",out)

		# dists=[]
		# for word in inference_params.quds:
		# 	subject = projection(ed.get_session().run(inference_params.listener_world),np.expand_dims(inference_params.vecs[word],-1))
		# 	observation = np.expand_dims(projection(np.mean(mus,axis=0),np.expand_dims(inference_params.vecs[word],-1)),0)
		# 	dists.append((word,np.linalg.norm(observation-subject)))
		# out2 = sorted(dists,key=lambda x:x[1],reverse=True)
		# print("L0 PROJECTION MOVEMENT\n",out2)

	if debug:
		print("tf mu", ed.get_session().run(mus),mus)
	# mus = tf.transpose(mus) #mus now contains a listener interpretation in each column
	
	projected_mus = tf.expand_dims(mus,0)
	projected_world = tf.expand_dims(tf.transpose(tf.expand_dims(s1_world,1)),0)

	# projected_mus = tf.transpose(tf.einsum('aij,jk->aik',qud_projection_matrix,mus),perm=[0,2,1])
	# projected_world = tf.transpose(tf.einsum('aij,jk->aik',qud_projection_matrix,tf.expand_dims(s1_world,1)),perm=[0,2,1])
	# print(ed.get_session().run(projected_mus[:,:,:5]),"PROJECTED MUS",projected_mus)
	# print("SHAPES",projected_mus.get_shape(),projected_world.get_shape())
	mus_world_diff = tf.transpose(tf.subtract(projected_mus,projected_world),perm=[0,2,1])
	rescaled_mu_diffs = tf.einsum('ij,ajk->aik',tf.sqrt(inference_params.inverse_sigma),mus_world_diff) #note that this is only correct if sigma is diagonal
	squared_diffs = tf.square(rescaled_mu_diffs)
	log_likelihoods = tf.multiply(-0.5,tf.reduce_sum(squared_diffs,axis=1))
	utterance_scores = tf.multiply(inference_params.rationality, tf.add(log_likelihoods,tf.multiply(inference_params.freq_weight,inference_params.weighted_utt_frequency_array)))
	#this is where the rationality parameter is included

	norm = tf.expand_dims(tf.reduce_logsumexp(utterance_scores,axis=1),1)
	out = tf.subtract(utterance_scores, norm)
	return out