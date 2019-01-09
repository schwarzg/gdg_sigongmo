# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 14:22:33 2019
[Tensorflow exercise] AND gate
@author: MSPL_Sein
"""

import numpy as np
import tensorflow as tf

#AND gate training data
x= np.array([[0,0],[0,1],[1,0],[1,1]]).astype('float32')
y_target= np.array([[0],[0],[0],[1]]).astype('float32')

w= tf.Variable(tf.zeros([2,1]))
b= tf.Variable(tf.zeros([1]))

#Hypothesis (modeling)
y= tf.nn.sigmoid(tf.matmul(x,w)+b)

#cost/ loss function
cost= -tf.reduce_sum(y_target*tf.log(y)+(1-y_target)*tf.log(1-y))

#Minimize
optimizer= tf.train.GradientDescentOptimizer(learning_rate=0.1)
train= optimizer.minimize(cost)

#Initializes global vaiables 
sess=tf.Session()
sess.run(tf.global_variables_initializer())

#Training
for epoch in range(200):
    sess.run(train)
    if epoch %20 == 0:
        print(epoch, sess.run(cost), sess.run(w), sess.run(b))

#Accuracy computation
correct_prediction= tf.equal(tf.to_float(tf.greater(y,0.5)),y_target)
classified= correct_prediction.eval(session=sess)
print(classified)
prob= y.eval(session=sess)
print(prob)