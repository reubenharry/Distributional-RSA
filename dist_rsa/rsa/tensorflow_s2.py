
def tf_s2(inference_params,s2_world):

	from dist_rsa.rsa.tensorflow_l1 import tf_l1
	import numpy as np
	import edward as ed
	import tensorflow as tf

	def s2_utility(u,i):
		print(i+1,"out of:",len(inference_params.possible_utterances))
		l1_scores = ed.get_session().run(tf_l1(inference_params)[0])
		# l1_quds = qud_combinations+[["TRIVIAL"]]
		# print(l1_quds)
		# print(q)
		# if q not in l1_quds:
		# 	ind = l1_quds.index([q[1],q[0]])
		# else:
		# 	ind = l1_quds.index(q)
		score = l1_scores[ind]

		# neg_score = l1_scores[l1_quds.index(['TRIVIAL'])]
		# neg_score = 0
		return score
		# ,neg_score

	# utilities = []
	# for i,utt in possible_utterances:
	# 	utilities.append()

	utilities = np.squeeze(np.asarray([s2_utility(utt,i)[0] for i,utt in enumerate(possible_utterances)]))

	# print(utilities.shape)
	uniform_utterance_prior = np.log(np.asarray([1/len(possible_utterances) for x in range(len(possible_utterances))]))
	# print(uniform_utterance_prior.shape,uniform_utterance_prior)
	posterior = utilities + uniform_utterance_prior
	#minimize trivial
	# rational_posterior = rationality * posterior
	# neg_utilities = np.squeeze(np.asarray([s2_utility(utt,i)[1] for i,utt in enumerate(possible_utterances)]))
	# posterior = uniform_utterance_prior - neg_utilities
	# check_posterior = np.exp(posterior)
	# print(posterior.shape,posterior)
	normalized_posterior = posterior - scipy.misc.logsumexp(posterior)
	# print(normalized_posterior.shape,normalized_posterior)
# 		self. list(zip(possible_utterances,normalized_posterior))
	# print(sorted(list(zip(possible_utterances,np.ndarray.tolist(np.exp(normalized_posterior)))),key=lambda x : x[1],reverse=True))
	return sorted(list(zip(possible_utterances,np.ndarray.tolist(np.exp(normalized_posterior)))),key=lambda x : x[1],reverse=True)
