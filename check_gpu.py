import tensorflow as tf
with tf.Session() as sess:
	#with tf.device("/gpu:0") as dev: #"/cpu:0" or "/gpu:0"
		tf.global_variables_initializer().run()
