import numpy as np
import nltk
import scipy
import pickle
from utils.helperfunctions import projection,weights_to_dist

def np_rsa(s1_world,qud_mat,vecs,vec_length, frequencies, world_mean, possible_utterances, utterance, lam=0,sig1=1.0,sig2=1.0,debug=True):
# np_rsa(s1_world,qud_mat,vecs,vec_length,sig1,sig2, frequencies, world_mean, possible_utterances, utterance, lam=0,debug=True)
	
	def l0(u,world_mean,sig1,sig2):
		n = 1
		y = u
		mu1 = world_mean
		sigma1 = sig1 ** 2
		sigma2 = sig1 ** 2
		mu = np.divide(np.add(mu1/sigma1, y/sigma2),  ((1/sigma1) + (1/sigma2)))
		sigma_base = ((1/sigma1) + (1/sigma2))**-1
		sigma = np.diag([sigma_base ** 2] * vec_length)
		return mu,sigma

	# return l0(u=utterance,world_mean=world_mean)

	def score_s1_utterance(world,qud,utterance,sig1,sig2,world_mean):
		# print(world_mean)
		print("world np",world,world.shape)
		print("np utterance", utterance,utterance.shape)
		listener_mu,listener_sigma = l0(u=utterance,world_mean=world_mean,sig1=sig1,sig2=sig2)
		print("mu np",listener_mu)
		# listener_mu,listener_sigma = l0(utterance,s1,s2,world)

		# print("np listener_mu",listener_mu)
		# print("qud matrix",qud)
		# print("NP WORLD",world)
		projected_world = projection(world,qud)
		print("np projected world",projected_world)
		projected_listener_mu = projection(listener_mu,qud)
		print("np projected mu",projected_listener_mu)
		projected_listener = scipy.stats.multivariate_normal(projected_listener_mu,listener_sigma)
		log_score = projected_listener.logpdf(projected_world)
		# return listener_mu
		# print("np log likelihood",log_score)
		raise Exception
		return log_score
		# a = (np.array([1.0,1.0]))
		# b = (np.diag(np.array([1.0,1.0])))
		# c = (np.array([0.5,0.5]))
		# # return tf.constant(5)
		# d =  scipy.stats.multivariate_normal(a,b)
		# return d.logpdf(c)

		# listener_mu,listener_sigma = l0(utterance,s1,s2,world_mean)
		# projected_world = projection(world,qud)
		# projected_listener_mu = projection(listener_mu,qud)
		


		# # projected_listener_mu = np.array([0.95188004, -0.64329422])
		# # projected_world = np.array([6.92073188,  -4.58426447])
		# # listener_sigma = np.diag([.25,.25])

		# # projected_listener_mu = (np.array([2.0,2.0]))
		# # projected_world = (np.array([0.5,0.5]))
		# # listener_sigma = np.diag([2.0,2.0])


		# projected_listener = scipy.stats.multivariate_normal(projected_listener_mu,listener_sigma)
		# # return "foo",projected_listener_mu, projected_world,listener_sigma,"foo"
		# log_score = projected_listener.logpdf(projected_world)
		# return log_score

	def s1(world,qud_mat,frequencies,possible_utterances,lam,sig1,sig2):
		world_mean = world
		world_vec = world

		# print(qud_mat.shape,world_vec.shape,vecs['predator'].shape)

		# world_vec = np.mean(world_vec.reshape(1,vecs[world].shape[0]) + (lam*qud_mat.T))

		# for i in range(len(qud_mat.T)):
		# 	world_vec = world_vec+lam*qud_mat.T[i] #shift the world_vec by the qud_vec
			
		utterances = possible_utterances

		freqs_log_scaled = np.log(np.array([(frequencies[x]) for x in utterances]))
		freq_dist = weights_to_dist(freqs_log_scaled)
		inp = zip(utterances,freq_dist)
		uniform_draw = [(vecs[x[0]],x[1]) for x in inp]
		
		# return [x[1] for x in uniform_draw]
		# return score_s1_utterance(world_vec,qud_mat,vecs['human'],s1,s2,world_mean) 

		norm = scipy.misc.logsumexp([x[1] + (score_s1_utterance(world_vec,qud_mat,x[0],sig1,sig2,world_mean)) for x in uniform_draw])
		# return norm
		out = [(x[1] + (score_s1_utterance(world_vec,qud_mat,x[0],sig1,sig2,world_mean))) - norm for x in uniform_draw]
		
		return out# output = list(zip(utterances,out))
		# output = [(x[0],np.exp(x[1])) for x in output]
		# return output

	#check if the score depends on the non-initial qud-words:
	out = s1(world=s1_world,qud_mat=qud_mat,frequencies=frequencies,possible_utterances=possible_utterances,lam=lam,sig1=sig1,sig2=sig2)
	# return sorted(out,key=lambda x: x[1],reverse=True)
	return out