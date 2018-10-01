from __future__ import division
import numpy as np
import scipy
import time
import itertools
import pickle
import numpy as np
import tensorflow as tf
from collections import defaultdict
from scipy.misc import logsumexp
from scipy.stats import rv_discrete
def orthogonal_complement_np(a):
	q,_ = np.linalg.qr(a,mode='complete')
	num_subspace_dims = a.shape[1]
	return q[:,num_subspace_dims:]

def orthogonal_complement_tf(a):
	q,_ = tf.linalg.qr(a,full_matrices=True)
	num_subspace_dims = a.get_shape()[1]
	return q[:,num_subspace_dims:]

def mean_and_variance_of_dist_array(probs,support):
	MEAN = tf.reduce_sum(probs*support,axis=0)
	mean_of_square_of_posterior = tf.reduce_sum(probs*(support**2),axis=0)
	VARIANCE = mean_of_square_of_posterior - tf.square(MEAN)
	return MEAN,VARIANCE

def mean_and_variance_of_dist_array_np(probs,support):

	# MEAN = np.sum(probs*support,axis=0)
	# mean_of_square_of_posterior = np.sum(probs*(support**2),axis=0)
	# VARIANCE = mean_of_square_of_posterior - np.square(MEAN)
	# print("old mean+var",MEAN,VARIANCE)
	# print(probs,np.exp(probs),"probs")
	dist = rv_discrete(values=(support,probs))
	# print("new mean+var",dist.mean(),dist.var())
	return dist.mean(),dist.var()

	# return MEAN,VARIANCE




# a = np.transpose(array([[1,0,0,0]]))

# a_orth = orthogonal_complement(a)

# print(np.dot(a[:,0],a_orth[:,0]))
# print(np.dot(a[:,0],a_orth[:,1]))

# print(a_orth)

#makes a matrix of glove vectors form a list of input words
def as_a_matrix(words,vecs):
	outs = np.asarray([vecs[x] for x in words])
	return outs

def lookup(dic,word,pos_tag):

	if dic[word] == 0 and dic[word+pos_tag]==0:
		return 1
	if dic[word] > 0:
		return dic[word]
	elif dic[word+pos_tag] > 0:
		return dic[word+'_'+pos_tag]


def combine_quds(quds,number_of_qud_dimensions):

	prod_quds = itertools.product(*([quds]*number_of_qud_dimensions))
	prod_quds = [sorted(q) for q in prod_quds]
	prod_quds = list(k for k,_ in itertools.groupby(sorted(prod_quds)))
	prod_quds = [q for q in prod_quds if len(q) == len(set(q))]
	return prod_quds

def projection(x,m):
	uncorrected_projection_weights = np.dot(np.transpose(m),x)
	covariance_matrix = np.dot(np.transpose(m),m)
	inverse_covariance_matrix = np.linalg.inv(covariance_matrix)
	projection_weights = np.dot(inverse_covariance_matrix,uncorrected_projection_weights)
	return np.dot(m,projection_weights)

#PROJECTS VECTORS INTO SUBSPACE, REPRESENTING OUTPUT IN SUBSPACE BASIS
# TYPE: [FULL_DIMS,NUM_VECS], [FULL_DIMS,SUBSPACE_DIMS] -> [NUM_VECS,SUBSPACE_DIMS] - I think
def projection_into_subspace_tf(x,m):
	uncorrected_projection_weights = tf.matmul(tf.transpose(m),x)
	covariance_matrix = tf.matmul(tf.transpose(m),m)
	inverse_covariance_matrix = tf.matrix_inverse(covariance_matrix)
	projection_weights = tf.matmul(inverse_covariance_matrix,uncorrected_projection_weights)
	return projection_weights

def projection_into_subspace_np(x,m):
	uncorrected_projection_weights = np.dot(np.transpose(m),x)
	covariance_matrix = np.dot(np.transpose(m),m)
	inverse_covariance_matrix = np.linalg.inv(covariance_matrix)
	projection_weights = np.dot(inverse_covariance_matrix,uncorrected_projection_weights)
	return projection_weights

def tensor_projection_matrix(m):
	covariance_matrix = tf.matmul(tf.transpose(m),m)
	inverse_covariance_matrix = tf.matrix_inverse(covariance_matrix)
	# print("NORMAL VERSION",m,inverse_covariance_matrix)
	return tf.matmul(tf.matmul(m,inverse_covariance_matrix),tf.transpose(m))

def double_tensor_projection_matrix(m):
	covariance_matrix = tf.einsum('aij,ajk->aik',tf.transpose(m,perm=[0,2,1]),m)
	inverse_covariance_matrix = tf.matrix_inverse(covariance_matrix)
	intermed = tf.einsum('aij,ajk->aik',m,inverse_covariance_matrix)
	return tf.einsum('aij,ajk->aik',intermed,tf.transpose(m,perm=[0,2,1]))

def double_tensor_projection_matrix_into_subspace(m):
	covariance_matrix = tf.einsum('aij,ajk->aik',tf.transpose(m,perm=[0,2,1]),m)
	inverse_covariance_matrix = tf.matrix_inverse(covariance_matrix)
	intermed = tf.einsum('aij,ajk->aik',m,inverse_covariance_matrix)
	return intermed





def projection_matrix(m):
	covariance_matrix = np.dot(np.transpose(m),m)
	inverse_covariance_matrix = np.linalg.inv(covariance_matrix)
	return np.dot(np.dot(m,inverse_covariance_matrix),m.T)

def normalize(w):
	w = np.array(w)
	norm = np.sum(w,axis=0)
	return w/norm


def weights_to_dist(weights):
	#assumes that weights is already log-scaled
	norm_constant = logsumexp(weights)
	dist = [(x - norm_constant) for x in weights]
	return dist

def tensor_projection(x,m):

	m = tf.transpose(m)
	x = tf.expand_dims(x,1)
	uncorrected_projection_weights = tf.matmul(m,x)
	m = tf.transpose(m)
	covariance_matrix = tf.matmul(tf.transpose(m),m)
	# inverse_covariance_matrix = tf.pow(tf.add(covariance_matrix,0.0000000001), -1)
	# sess = tf.Session()
	# print(sess.run(covariance_matrix).tolist())
	inverse_covariance_matrix = tf.matrix_inverse(covariance_matrix)
	projection_weights = tf.matmul(inverse_covariance_matrix,uncorrected_projection_weights)
	return tf.reshape(tf.matmul(m,projection_weights),[-1])

def unique_list(seq):
	seen = set()
	seen_add = seen.add
	return [x for x in seq if not (x in seen or seen_add(x))]

def s1_nonvect(speaker_world,qud_mat,lam=0.0,freq_weight=1.0):

	speaker_world = tf.add(speaker_world,tf.multiply(lam,tf.reduce_sum(tf.transpose(qud_mat),axis=0)))
	utterances = possible_utterances
	qud_projection_matrix = tensor_projection_matrix(qud_mat)
	mus = tf.divide(tf.add(listener_world/sigma1, poss_utts/sigma2),  inverse_sd)
	mus = tf.transpose(mus) #mus now contains a listener interpretation in each column
	projected_mus = tf.transpose(tf.matmul(qud_projection_matrix,mus))
	projected_world = tf.transpose(tf.matmul(qud_projection_matrix,tf.expand_dims(speaker_world,1)))
	mus_world_diff = tf.transpose(tf.subtract(projected_mus,projected_world))
	rescaled_mu_diffs = tf.matmul(tf.sqrt(inverse_sigma),mus_world_diff) #note that this is only correct if sigma is diagonal
	squared_diffs = tf.square(rescaled_mu_diffs)
	log_likelihoods = tf.multiply(-0.5,tf.reduce_sum(squared_diffs,axis=0))
	utterance_scores = tf.add(log_likelihoods,tf.multiply(freq_weight,weighted_utt_frequency_array))
	norm = tf.reduce_logsumexp(utterance_scores)
	out = tf.subtract(utterance_scores, norm)
	return out

def memoized_s2(q,utt_to_qud_dist):
		possible_utterances = list(utt_to_qud_dist)

		def s2_utility(u,i):
			l1_quds,l1_scores = list(zip(*utt_to_qud_dist[u]))
			sorted_q = sorted(q)
			# print(l1_quds)
			# print(sorted_q)
			# print(i)
			ind = l1_quds.index(sorted_q)
			score = l1_scores[ind]

			# neg_score = l1_scores[l1_quds.index(['TRIVIAL'])]
			neg_score = 0
			return score,neg_score

		utilities = np.squeeze(np.asarray([s2_utility(utt,i)[0] for i,utt in enumerate(possible_utterances)]))
		uniform_utterance_prior = np.log(np.asarray([1/len(possible_utterances) for x in range(len(possible_utterances))]))
		posterior = utilities + uniform_utterance_prior
		# rational_posterior = rationality * posterior
		# neg_utilities = np.squeeze(np.asarray([s2_utility(utt,i)[1] for i,utt in enumerate(possible_utterances)]))

		normalized_posterior = posterior - logsumexp(posterior)
		out = sorted(list(zip(possible_utterances,np.ndarray.tolist(((normalized_posterior))))),key=lambda x : x[1],reverse=True)
		return out

def metaphors_to_csv(metaphors,l1,baseline,controls):
	import csv
	import random
	from dist_rsa.utils.helperfunctions import unique_list

	items = []
	# random.shuffle(metaphors)
	for sentence,metaphor in metaphors:
		b = bool(random.getrandbits(1))
		l1s = unique_list(l1[metaphor])
		baselines = unique_list(baseline[metaphor])
		for i in range(3):
			adj1 = l1s[i]
			adj2 = baselines[i]
			if b:
				meta=adj1
				metb=adj2
			else:
				metb=adj1
				meta=adj2
			items.append([sentence,meta,metb,adj1,str(i)])

	for control in controls:
		items.append(control)

	random.shuffle(items)

	with open('mturk.csv', 'w',newline='') as csvfile:
		csv_writer = csv.writer(csvfile, delimiter=',',
		quotechar='|', quoting=csv.QUOTE_MINIMAL)
		csv_writer.writerow(["SENTENCE","ADJ1","ADJ2","L1","RANK"])

		for item in items:
			csv_writer.writerow(item)

def display(full_name):
    from dist_rsa.utils.load_data import metaphors
    l1 = pickle.load(open("dist_rsa/data/l1_dict"+full_name,'rb'))
    baseline = pickle.load(open("dist_rsa/data/baseline_dictshort_2",'rb'))

    for metaphor in metaphors:
        metaphor = tuple(metaphor)
        print('\n\n\n',metaphor)
        print("BASELINE:",baseline[metaphor])
        print("L1:",l1[metaphor])

def metaphors_to_html(metaphors,l1,baseline,controls,return_dict=False):
	import csv
	import random
	from dist_rsa.utils.helperfunctions import unique_list

	items = []
	# random.shuffle(metaphors)
	for sentence,metaphor in metaphors:
		b = bool(random.getrandbits(1))
		l1s = unique_list(l1[metaphor])
		baselines = unique_list(baseline[metaphor])
		for i in range(3):
			adj1 = l1s[i]
			adj2 = baselines[i]
			if b:
				meta=adj1
				metb=adj2
			else:
				metb=adj1
				meta=adj2
			items.append([sentence,meta,metb,adj2,str(i)])

	for control in controls:
		items.append(control)

	random.shuffle(items)

	html_text=''

	if return_dict:
		out = []
		for i,item in enumerate(items):
			out+=[item[3]]
		return out

	for i,item in enumerate(items):

		# html_text+='<div class="panel-body"><label>'+item[0]+'</label><div class="radio"><label><input name="adj1" required="" type="radio" value="" />'+item[1]+'</label></div><div class="radio"><input name="adj2" required="" type="radio" value="" />'+item[2]+'</div></div>'
		html_text+='<p>'+item[0]+':&nbsp;<input name="adj'+str(i)+'" type="radio" value="a1" />&nbsp;'+item[1]+' &nbsp;<input name="adj'+str(i)+'" type="radio" value="a2" />&nbsp;'+item[2]+'</p>'
		# item[0]+'<input name="adj'+str(i)+'" type="radio" value="a1" />'+item[1]+'<input name="adj'+str(i)+'" type="radio" value="a2" />'+item[2]+'</section>'
	html_file = open("html_file",'w')
	html_file.write(html_text)
	html_file.close()

def demarginalize_product_space(product_dist):

	"""
	product_dist is a list of tuples of (n-tuple of words,probability mass of that n-tuple)
	"""

	import numpy as np
	from collections import defaultdict

	#calcuate number of dimensions of distribution's support
	size_of_product = len(product_dist[0])

	# unzip product_dist into the list of words, and the list of scores 
	product_dist_words,product_dist_scores = zip(*product_dist)

	# obtain the total set of words by flattening the list of n-tuples and removing duplicates
	words = set([item for sublist in product_dist_words for item in sublist])

	#initialize a dictionary whose default value for any key whatsoever is 0.0
	word_scores=defaultdict(float)

	# loop over words, and for each, loop over product_dist, adding the score of each tuple it appears in
	for word in list(words):
		for item in product_dist:
			if word in item[0]:
				word_scores[word]+=np.exp(item[1])
	# convert word_scores dictionary to list of tuples and sort by the size of the second element of each tuple, in reverse order
	word_scores_list = sorted(list(word_scores.items()),key=lambda x:x[1],reverse=True)

	# unzip word_scores_list (which is list of tuples) into two lists, of words, and scores
	final_words,scores = zip(*word_scores_list)

	# normalize distribution
	scores = np.ndarray.tolist(np.array(scores)/np.sum(scores))
	return list(zip(final_words,scores))
