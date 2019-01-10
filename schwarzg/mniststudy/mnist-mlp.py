import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("./",one_hot=True)

#28*28 pixel image -> 10 digits
X=tf.placeholder(tf.float32,shape=[None,784])
Y=tf.placeholder(tf.float32,shape=[None,10])

#one hidden layer
W1=tf.Variable(tf.random_normal([784,500]))
b1=tf.Variable(tf.random_normal([500]))

H=tf.nn.sigmoid(tf.add(tf.matmul(X,W1),b1))

W2=tf.Variable(tf.random_normal([500,10]))
b2=tf.Variable(tf.random_normal([10]))

y_conv=tf.add(tf.matmul(H,W2),b2)
actv=tf.nn.sigmoid(tf.add(tf.matmul(H,W2),b2))


cross_entropy=Y*tf.log(actv)+(1.0-Y)*tf.log(1.0-actv)

#cost=tf.reduce_mean(-tf.reduce_sum(cross_entropy,reduction_indices=[1]))

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=y_conv))

learning_rate=0.001

optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10000):
	batch_xs,batch_ys = mnist.train.next_batch(600)
	sess.run(optimizer,feed_dict={X:batch_xs,Y:batch_ys})
	if i%100==0:
		correct_prediction=tf.equal(tf.argmax(actv,1),tf.argmax(Y,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
		print(i, sess.run(cost,feed_dict={X:batch_xs,Y:batch_ys}), sess.run(accuracy,feed_dict={X:batch_xs,Y:batch_ys}))

correct_prediction=tf.equal(tf.argmax(actv,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print(sess.run(accuracy,feed_dict={X:mnist.test.images,Y:mnist.test.labels}))
