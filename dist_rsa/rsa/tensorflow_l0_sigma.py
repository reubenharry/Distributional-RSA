
def tf_l0_sigma(inference_params):
	import tensorflow as tf
	sigma1 = inference_params.sig1 ** 2
	sigma2 = inference_params.sig2 ** 2
	inverse_sd = (1/sigma1) + (1/sigma2)
	sigma = tf.diag([tf.pow(inverse_sd,-1)] * inference_params.vec_length)
	inverse_sigma = tf.matrix_inverse(sigma)
	# print("sigma info",sigma1,sigma2,inverse_sd,sigma)
	return sigma1,sigma2,inverse_sd,sigma,inverse_sigma