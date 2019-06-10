
mu = 0.0
sigma = 1.0

def gaussian_likelihood_unvectorized_built_in(mu,sigma,observation):

	return scipy.stats.multivariate_normal(mu,sigma).logpdf(observation)


def gaussian_likelihood_unvectorized_handmade(mu,sigma,observation):

	pass


# shape
def gaussian_likelihood_vectorized(mus,sigma,observations):

	pass


import scipy
from dist_rsa.utils.load_data import load_vecs
import numpy as np
vec_size,vec_kind = 25,'glove.twitter.27B.'
vecs = load_vecs(mean=True,pca=True,vec_length=vec_size,vec_type=vec_kind)  


a = vecs['the']

b = vecs['a']
c = vecs['cat']

tens = np.array([b,c]).T 

print(tens.shape)

def generate_orthonomal_basis(tens):
	return scipy.linalg.orth(tens)
	
def project(obj,proj):
	mat = generate_orthonomal_basis(proj)
	return np.dot(obj,mat)

print(project(a,tens),project(a,tens).shape)

# scipy.linalg




# print(gaussian_likelihood_unvectorized_built_in(mu,sigma,))