import numpy as np
import tensorflow as tf
from dist_rsa.utils.helperfunctions import projection_into_subspace_tf

a = np.random.rand(3,1)
b = np.random.rand(3,1)
c = np.zeros((3,1))
c[2,:]=1
# c = np.random.rand(30,1)
# d = np.random.rand(30,1)
sess = tf.Session()

print(c)
# WITH LOOP
out = []
for x in [a,b]:
	out.append(projection_into_subspace_tf(x,c))

print(out)
loop1 = sess.run(tf.concat(out,axis=1))
print(loop1,"LOOP1")

# VECTORIZED
vectorized1 = sess.run(projection_into_subspace_tf(tf.concat([a,b],axis=1),c))
print(vectorized1,"VECTORIZED1")


assert(np.array_equal(loop1,vectorized1))


out = []
for x in [a,b]:
	out.append(projection_into_subspace_tf(c,x))

print(out)
loop2 = sess.run(tf.concat(out,axis=1))
print(loop2,"LOOP2")

# VECTORIZED
vectorized2 = sess.run(projection_into_subspace_tf(c,tf.concat([a,b],axis=1)))
print(vectorized2,"VECTORIZED2")

# assert(np.array_equal(loop2,vectorized2))



