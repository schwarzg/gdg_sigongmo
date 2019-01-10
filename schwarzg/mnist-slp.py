import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("./",one_hot=True)

#28*28 pixel image -> 10 digits
X=tf.placeholder(tf.float32,shape=[None,784])
Y=tf.placeholder(tf.float32,shape=[None,10])

W=tf.Variable(tf.random_normal([784,10]))
b=tf.Variable(tf.random_normal([10]))

z=tf.matmul(X,W)+b

actv=tf.nn.softmax(z)

cross_entropy=-tf.reduce_sum(Y*tf.log(actv))

learning_rate=0.01

optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
	batch_xs,batch_ys = mnist.train.next_batch(100)
	sess.run(optimizer,feed_dict={X:batch_xs,Y:batch_ys})

	correct_prediction=tf.equal(tf.argmax(actv,1),tf.argmax(Y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

	print(sess.run(accuracy,feed_dict={X:mnist.test.images,Y:mnist.test.labels}))
