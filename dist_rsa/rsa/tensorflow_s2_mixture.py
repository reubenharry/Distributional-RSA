
def tf_s2_mixture(inference_params,s2_qud):

	from dist_rsa.rsa.tensorflow_l1_mixture import tf_l1
	import numpy as np
	# import edward as ed
	# import tensorflow as tf
	import scipy

	# sess = ed.get_session()


	def s2_utility(u):
		inference_params.predicate=u
		l1_scores = tf_l1(inference_params)
		vals,probs = zip(*l1_scores)
		# print("probs",probs)
		# l1_quds = qud_combinations+[["TRIVIAL"]]
		# print(l1_quds)
		print("results",l1_scores[:10])
		# print(q)
		# if q not in l1_quds:
		# 	ind = l1_quds.index([q[1],q[0]])
		# else:
		# 	ind = l1_quds.index(q)
		score = probs[vals.index(s2_qud)]
		print("UTTERANCE AND SCORE",u,score)

		# neg_score = l1_scores[l1_quds.index(['TRIVIAL'])]
		# neg_score = 0
		return score
		# ,neg_score


	utilities = np.squeeze(np.asarray([s2_utility(utt) for _,utt in enumerate(inference_params.possible_utterances)]))

	# print(utilities.shape)
	uniform_utterance_prior = np.log(np.asarray([1/len(inference_params.possible_utterances) for x in range(len(inference_params.possible_utterances))]))
	# print(uniform_utterance_prior.shape,uniform_utterance_prior)
	posterior = utilities + uniform_utterance_prior
	#minimize trivial
	# rational_posterior = rationality * posterior
	# neg_utilities = np.squeeze(np.asarray([s2_utility(utt,i)[1] for i,utt in enumerate(inference_params.possible_utterances)]))
	# posterior = uniform_utterance_prior - neg_utilities
	# check_posterior = np.exp(posterior)
	# print(posterior.shape,posterior)
	normalized_posterior = posterior - scipy.misc.logsumexp(posterior)
	# print(normalized_posterior.shape,normalized_posterior)
# 		self. list(zip(inference_params.possible_utterances,normalized_posterior))
	# print(sorted(list(zip(inference_params.possible_utterances,np.ndarray.tolist(np.exp(normalized_posterior)))),key=lambda x : x[1],reverse=True))
	out = sorted(list(zip(inference_params.possible_utterances,np.ndarray.tolist(np.exp(normalized_posterior)))),key=lambda x : x[1],reverse=True)
	print(out)
	raise Exception
	return out