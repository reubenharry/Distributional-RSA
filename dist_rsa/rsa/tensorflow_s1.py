# if you're not comfortable with einstein summation, this vectorized implementation of the s1 may appear inscrutable. 
# For the semantics of the s1, try looking at the numpy version instead (numpy_rsa.py)


def tf_s1(inference_params,s1_world,world_movement=False,debug=False,NUMPY=False,):

	"""
	s1_world: SHAPE: [1, NUM_DIMS]


	"""



	import tensorflow as tf
	import edward as ed
	import numpy as np
	import scipy
	from dist_rsa.utils.helperfunctions import projection,tensor_projection,as_a_matrix,double_tensor_projection_matrix,combine_quds, lookup
	from dist_rsa.rsa.tensorflow_l0_sigma import tf_l0_sigma

	if NUMPY:
		qud_matrix = tf.cast(inference_params.qud_matrix,dtype=tf.float32)
		listener_world = tf.cast(inference_params.listener_world,dtype=tf.float32)
		s1_world = tf.cast(s1_world,dtype=tf.float32)
	else:
		qud_matrix = inference_params.qud_matrix
		listener_world = inference_params.listener_world



	inference_params.sigma1,inference_params.sigma2,inference_params.inverse_sd,inference_params.sigma,inference_params.inverse_sigma = tf_l0_sigma(inference_params)
	# print(inference_params.listener_world.get_shape(),inference_params.sigma1.get_shape(),inference_params.poss_utts.get_shape(),inference_params.sigma2.get_shape())

	# SHAPE: [NUM_QUDS,NUM_DIMS,NUM_DIMS] ?
	qud_projection_matrix = double_tensor_projection_matrix(qud_matrix)
	# SHAPE: [NUM_UTTS,NUM_DIMS]
	mus = tf.divide(tf.add(listener_world/inference_params.sigma1, 
		inference_params.poss_utts/inference_params.sigma2),inference_params.inverse_sd)


	if world_movement:
		pass
		# print("LISTENER MUS SHAPE",mus.shape)
		words_with_distance_to_prior = np.asarray(list(map(lambda x: scipy.spatial.distance.cosine(inference_params.vecs[x],ed.get_session().run(listener_world)),inference_params.quds)))
		words_with_distance_to_posterior = np.asarray(list(map((lambda y:   list(map((lambda x: scipy.spatial.distance.cosine(inference_params.vecs[x],y)),inference_params.quds))),ed.get_session().run(mus))))
		mean_word_movement = np.expand_dims(np.mean(words_with_distance_to_posterior-words_with_distance_to_prior,axis=0),0)
		out = sorted(list(zip(inference_params.quds,mean_word_movement)),key=lambda x:x[1],reverse=True)
		print("L0 world movement:",out)
		# for x in word_movement:
		for i in range(mus.get_shape()[0]):
			dists=[]
			for word in inference_params.quds:
				subject = projection(ed.get_session().run(listener_world),np.expand_dims(inference_params.vecs[word],-1))
				observation = np.expand_dims(projection(ed.get_session().run(mus[i]),np.expand_dims(inference_params.vecs[word],-1)),0)
				dists.append((word,np.linalg.norm(observation-subject)))
			out2 = sorted(dists,key=lambda x:x[1],reverse=True)
			print("L0 PROJECTION MOVEMENT\n",out2)

	# print("TF WORLD",ed.get_session().run(s1_world))
	# print("tf mu", ed.get_session().run(mus),mus)
		# print("tf qud_projection_matrix",qud_projection_matrix,ed.get_session().run(qud_projection_matrix[0][0]))
	if debug:
		pass
		# print("s1 world", s1_world,ed.get_session().run(s1_world))
		# print("tf listener prior mean",ed.get_session().run(inference_params.listener_world))
		# print("possible utts",inference_params.poss_utts,ed.get_session().run(inference_params.poss_utts))
		# print("mus",mus,ed.get_session().run(mus))
	s1_world = tf.transpose(s1_world)
	# SHAPE: [NUM_DIMS,NUM_UTTS]
	transposed_mus = tf.transpose(mus) #mus now contains a listener interpretation in each column
	# SHAPE: [NUM_QUDS,NUM_UTTS,NUM_DIMS]
	projected_mus = tf.transpose(tf.einsum('aij,jk->aik',qud_projection_matrix,transposed_mus),perm=[0,2,1])
	# SHAPE: [NUM_QUDS,1,NUM_DIMS]
	# print(qud_projection_matrix)
	# print(s1_world)
	projected_world = tf.transpose(tf.einsum('aij,jk->aik',qud_projection_matrix,s1_world),perm=[0,2,1])
	# print("new projected world",ed.get_session().run(tf.shape(projected_world)),"new projected mus",ed.get_session().run(tf.shape(projected_mus)))
	# inference_params.mus=mus
	# inference_params.projected_mus=projected_mus

	# print(projected_mus.get_shape(),projected_world.get_shape())

	if debug:
		pass
		# print("tf projected_world dv t", ed.get_session().run(projected_world[:,:,:]),projected_world)
		# print("PROJECTED MUS dv t",ed.get_session().run(projected_mus),projected_mus)
	
	# stacked_unprojected_world = tf.transpose(tf.stack([s1_world,s1_world,s1_world]),perm=[0,2,1])
	# stacked_unprojected_mus = tf.transpose(tf.stack([mus,mus,mus]),perm=[0,2,1])
	
	# print("stackeds",ed.get_session().run([stacked_unprojected_mus,stacked_unprojected_world]))

	# SHAPE: [NUM_QUDS,NUM_UTTS,NUM_DIMS]
	mus_world_diff = tf.transpose(tf.subtract(projected_mus,projected_world),perm=[0,2,1])

	# mus_world_diff = tf.transpose(tf.subtract(stacked_unprojected_mus,stacked_unprojected_world),perm=[0,2,1])
	if debug:
		pass
		# print("mus_world_diff double vect",mus_world_diff,ed.get_session().run(mus_world_diff))
	rescaled_mu_diffs = tf.einsum('ij,ajk->aik',tf.sqrt(inference_params.inverse_sigma),mus_world_diff) #note that this is only correct if sigma is diagonal
	squared_diffs = tf.square(rescaled_mu_diffs)
	log_likelihoods = tf.multiply(-0.5,tf.reduce_sum(squared_diffs,axis=1))
	
	utterance_scores = tf.multiply(inference_params.rationality, tf.add(log_likelihoods,tf.multiply(inference_params.freq_weight,inference_params.weighted_utt_frequency_array)))
	if debug:
		# pass
		# print("RATIONALITY",inference_params.rationality)
		print("tf log_likelihoods",ed.get_session().run(log_likelihoods))
		# print("tf scores",ed.get_session().run(utterance_scores))
	#this is where the rationality parameter is included
	norm = tf.expand_dims(tf.reduce_logsumexp(utterance_scores,axis=1),1)
	out = tf.subtract(utterance_scores, norm)

	# print(out.get_shape(),"shape out")
	return out