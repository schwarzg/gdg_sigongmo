"""
Created on Tue Jan  8 14:22:33 2019
[Tensorflow exercise] OR gate
@author: MSPL_Sein
"""
import numpy as np
import tensorflow as tf
tf.set_random_seed(777) #for reproducibility

X= tf.placeholder(tf.float32, shape=[None, 2])
Y_target= tf.placeholder(tf.float32, shape=[None, 1])

w= tf.Variable(tf.random_normal([2,1]), name='weight')
b= tf.Variable(tf.random_normal([1]), name='bias')

#Hypothesis (modeling)
y= tf.sigmoid(tf.matmul(X,w)+b)

#cost/ loss function
cost= -tf.reduce_sum(Y_target*tf.log(y)+(1-Y_target)*tf.log(1-y))

#Minimize
optimizer= tf.train.GradientDescentOptimizer(learning_rate=0.1)
train= optimizer.minimize(cost)

#OR gate training data
x=np.array( [[0,0],[0,1],[1,0],[1,1]] )
y_target=np.array( [[0],[1],[1],[1]] )

#Initializes global vaiables 
sess=tf.Session()
sess.run(tf.global_variables_initializer())

#Training
for epoch in range(200):
    sess.run(train, feed_dict={
            X: x, 
            Y_target: y_target
            })
    if epoch %20 == 0:
        Cost,Weight,Bias= sess.run([cost,w,b],feed_dict={X: x, Y_target: y_target})
        print("\nCost: ",Cost,"\nWeight: ",Weight,"\nBias: ",Bias)

#Accuracy computation
correct_prediction= tf.equal(tf.to_float(tf.greater(y,0.5)),y_target)
classified= correct_prediction.eval(session=sess,feed_dict={X: x, Y_target: y_target})
print(classified)
prob= y.eval(session=sess,feed_dict={X: x, Y_target: y_target})
print(prob)