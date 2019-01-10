import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("./",one_hot=True)

def weight_variable(shape):
	initial=tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial=tf.constant(0.1,shape=shape)
	return tf.Variable(initial)

#convolution layer
def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#pooling layer
def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


X=tf.placeholder(tf.float32,shape=[None,784])
Y=tf.placeholder(tf.float32,shape=[None,10])

W_c1=weight_variable([5,5,1,32])	#32 features of 5x5 patches with 1 input channel
b_c1=bias_variable([32])

x_image=tf.reshape(X,[-1,28,28,1]) 	#number of image, x, y, channel

h_c1=tf.nn.relu(conv2d(x_image,W_c1)+b_c1)
h_p1=max_pool_2x2(h_c1)

W_c2=weight_variable([5,5,32,64])	#64 features of 5x5 patches with 32 input channel
b_c2=bias_variable([64])

h_c2=tf.nn.relu(conv2d(h_p1,W_c2)+b_c2)
h_p2=max_pool_2x2(h_c2)

W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])

h_pool2_flat=tf.reshape(h_p2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])

y_conv=tf.matmul(h_fc1_drop,W_fc2)+b_fc2

cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=y_conv))

train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

sess=tf.Session()

sess.run(tf.global_variables_initializer())
for i in range(20000):
	batch=mnist.train.next_batch(50)
	if i%100==0:
		train_accuracy=sess.run(accuracy,feed_dict={X:batch[0],Y:batch[1],keep_prob:0.5})
		print('step %d, training accuracy %g'%(i,train_accuracy))
	sess.run(train_step,feed_dict={X:batch[0],Y:batch[1],keep_prob:1.0})

print 'test accuracy %g'%sess.run(accuracy,feed_dict={X:mnist.test.images,Y:mnist.test.labels,keep_prob:1.0})
