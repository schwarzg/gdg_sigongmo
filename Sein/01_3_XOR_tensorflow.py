# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 10:37:43 2019
[Tensorflow exercise] XOR gate
@author: MSPL
"""
import tensorflow as tf
tf.set_random_seed(777)

x=tf.placeholder(tf.float32, shape=[None,2])
y_target=tf.placeholder(tf.float32, shape=[None,1])

#Hypothesis (modeling)
w1=tf.Variable(tf.random_normal([2,2]), name='weight1')
b1=tf.Variable(tf.random_normal([2]), name='bias1')
layer1= tf.nn.sigmoid(tf.matmul(x,w1)+b1)

w2=tf.Variable(tf.random_normal([2,1]),name='weight2')
b2=tf.Variable(tf.random_normal([1]),name='bias2')
y= tf.nn.sigmoid(tf.matmul(layer1,w2)+b2)

#cost/ loss function
cost= -tf.reduce_sum(y_target*tf.log(y)+(1-y_target)*tf.log(1-y))

#Minimize
optimizer= tf.train.GradientDescentOptimizer(learning_rate=0.06)
train= optimizer.minimize(cost)

"모델 학습"
#XOR gate training data
X=[[0,0], [0,1], [1,0], [1,1]]
Y_target= [[0], [1], [1], [0]]

#Initializes global vaiables 
sess=tf.Session()
sess.run(tf.global_variables_initializer())

#Training
for epoch in range(2000):
    sess.run(train, feed_dict={
            x: X, 
            y_target: Y_target
            })
    if epoch %20 == 0:
        Cost,Weight,Bias= sess.run([cost,w2,b2],feed_dict={x: X, y_target: Y_target})
        print("\nCost: ",Cost,"\nWeight: ",Weight,"\nBias: ",Bias)

#Accuracy computation
correct_prediction= tf.equal(tf.to_float(tf.greater(y,0.5)),y_target)
classified= correct_prediction.eval(session=sess,feed_dict={x: X, y_target: Y_target})
print("\n", classified)
prob= y.eval(session=sess,feed_dict={x: X, y_target: Y_target})
print(prob)