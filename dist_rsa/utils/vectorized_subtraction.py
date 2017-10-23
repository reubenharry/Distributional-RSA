import tensorflow as tf

sess = tf.Session()

a = tf.random_uniform((2,3,4))
b = tf.random_uniform((2,1,4))
c = tf.random_uniform((2,1,4))
d = tf.concat([b,c],axis=1)

print(d.get_shape())
# print(b.get_shape(),c.get_shape(),d[:,0,:].get_shape(),d[:,1,:].get_shape())
# print(sess.run([b,c,d[:,0,:],d[:,1,:]]))

def fast_vectorized_subtraction(u,v):
	u2 = tf.expand_dims(u,1)
	v2 = tf.expand_dims(v,2)
	return u2-v2

def slow_vectorized_subtraction(u,v):
	v2 = tf.expand_dims(v[:,0,:],1)
	print(v2.get_shape())
	v3 = tf.expand_dims(v[:,1,:],1)
	print(v3.get_shape())
	return tf.concat([a-v2,a-v3],axis=1)

print(fast_vectorized_subtraction(a,d).get_shape())

# print(sess.run([fast_vectorized_subtraction(a,d),slow_vectorized_subtraction(a,d)]))