'''
###########################################################
GDG RNN STUDY HW#1
Name : Chang Hyun Lee
Date : 2019.1.21.
###########################################################
Content : XOR gate using MLP - tensorflow
###########################################################
'''



import numpy as np
import tensorflow as tf

tf.set_random_seed(0)

X=np.array([[0,0],[0,1],[1,0],[1,1]])
Y=np.array([[0],[1],[1],[0]])

x=tf.placeholder(tf.float32, shape=[None, 2])
t=tf.placeholder(tf.float32, shape=[None, 1])

W= tf.Variable(tf.zeros([2, 2]))
b= tf.Variable(tf.zeros([2]))
h=tf.nn.sigmoid(tf.matmul(x,W)+b)

V=tf.Variable(tf.truncated_normal([2,1]))
c=tf.Variable(tf.zeros([1]))
y=tf.nn.sigmoid(tf.matmul(h,V)+c)

cross_entropy = -tf.reduce_sum(t*tf.log(y)+(1-t)*tf.log(1-y))

train_step=tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
correct_prediction