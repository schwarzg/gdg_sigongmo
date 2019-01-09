# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 17:15:10 2019
* Multi-class classification- input(2)/ output(3)
* mini-batch training
@author: MSPL
"""

import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
from sklearn.utils import shuffle

# data 생성
M= 2 # input data dimension
K= 3 # class 수
n= 100 # 각 class의 data 개수
N= n*K

X1= np.random.randn(n,M)+np.array([0,10])
X2= np.random.randn(n,M)+np.array([5,5])
X3= np.random.randn(n,M)+np.array([10,0])
Y1= np.array([[1,0,0] for i in range(n)])
Y2= np.array([[0,1,0] for i in range(n)])
Y3= np.array([[0,0,1] for i in range(n)])

X= np.concatenate((X1,X2,X3), axis=0)
Y= np.concatenate((Y1,Y2,Y3), axis=0)
plt.scatter(X[:,0], X[:,1])
plt.show()

#Hypothesis (modeling)
x=tf.placeholder(tf.float32, shape=[None,M])
y_target=tf.placeholder(tf.float32, shape=[None,K])
w=tf.Variable(tf.zeros([M,K]), name='weight')
b=tf.Variable(tf.zeros([K]), name='bias')
#w=tf.Variable(tf.random_normal([M,K]), name='weight')
#b=tf.Variable(tf.random_normal([K]), name='bias')
y=tf.nn.softmax(tf.matmul(x,w)+b)

#cost/ loss function
cost= tf.reduce_mean(-tf.reduce_sum(y_target*tf.log(y),reduction_indices=[1]))

#Minimize
optimizer= tf.train.GradientDescentOptimizer(learning_rate=0.1)
train= optimizer.minimize(cost) 

"모델 학습"
#Initializes global vaiables 
sess=tf.Session()
sess.run(tf.global_variables_initializer())

#Training
batch_size= 50
for epoch in range(20):
    X_, Y_ =shuffle(X,Y)
    for i in range(N):
        start= i * batch_size
        end= start + batch_size
        sess.run(train, feed_dict={
            x: X_[start:end], 
            y_target: Y_[start:end]
            })
    if epoch %2 == 0:
        Cost,Weight,Bias= sess.run([cost,w,b],feed_dict={x: X_[0:10],y_target: Y_[0:10]})
        print("\nCost: ",Cost,"\nWeight: ",Weight,"\nBias: ",Bias) 
    
#Accuracy computation (Testing)
X_, Y_ =shuffle(X,Y)
correct_prediction= tf.equal(tf.argmax(y,1),tf.argmax(y_target,1))
classified= correct_prediction.eval(session=sess,feed_dict={x: X_[0:10],y_target: Y_[0:10]})
print("\n", classified)
prob= y.eval(session=sess,feed_dict={x: X_[0:10]})
print(prob)