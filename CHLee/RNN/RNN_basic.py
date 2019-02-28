import tensorflow as tf
import numpy as np

n_inputs = 3
n_neurons = 5

X0 = tf.placeholder(tf.float32,[None, n_inputs])
X1 = tf.placeholder(tf.float32,[None, n_inputs])



'''
###
텐서플로로 기본 RNN 구성하기
###
Wx=tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons],dtype=tf.float32))
Wy=tf.Variable(tf.random_normal(shape=[n_neurons, n_neurons],dtype=tf.float32))
b=tf.Variable(tf.zeros([1, n_neurons],dtype=tf.float32))


Y0=tf.tanh(tf.matmul(X0,Wx)+b)
Y1=tf.tanh(tf.matmul(Y0,Wy)+tf.matmul(X1,Wx)+b)

init = tf.global_variables_initializer()

print(X0)

X0_batch=np.array([[0,1,2],[3,4,5],[6,7,8],[9,0,1]]) # t=0 일 때
X1_batch=np.array([[9,8,7],[0,0,0],[6,5,4],[3,2,1]]) # t=1 일 때

print(X0_batch)

with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0,Y1],feed_dict={X0:X0_batch, X1:X1_batch})

print(Y0_val)
print(Y1_val)


'''

'''
###
정적으로 타임 스텝 펼치기
###
'''

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell,[X0,X1],dtype=tf.float32)

Y0, Y1=output_seqs

init = tf.global_variables_initializer()

X0_batch=np.array([[0,1,2],[3,4,5],[6,7,8],[9,0,1]]) # t=0 일 때
X1_batch=np.array([[9,8,7],[0,0,0],[6,5,4],[3,2,1]]) # t=1 일 때

with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0:X0_batch, X1:X1_batch})

print(Y0_val)
print(Y1_val)
