'''
###########################################################
GDG RNN STUDY HW#1
Name : Chang Hyun Lee
Date : 2019.1.9.
###########################################################
Content : AND,OR,XOR gate - tensorflow
###########################################################
'''


import numpy as np
import tensorflow as tf

tf.set_random_seed(0)

w= tf.Variable(tf.zeros([2, 1]))
b= tf.Variable(tf.zeros([1]))

x=tf.placeholder(tf.float32, shape=[None, 2])
t=tf.placeholder(tf.float32, shape=[None, 1])
y=tf.nn.sigmoid(tf.matmul(x,w)+b)

cross_entropy = - tf.reduce_sum(t*tf.log(y)+(1-t)*tf.log(1-y))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

correct_prediction = tf.equal(tf.to_float(tf.greater(y,0.5)),t)


X=np.array([[0,0],[0,1],[1,0],[1,1]])
Y=np.array([[0],[0],[0],[1]]) #AND
#Y=np.array([[0],[1],[1],[1]]) #OR
#Y=np.array([[0],[1],[1],[0]]) #XOR

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(200):
    sess.run(train_step, feed_dict={
        x:X,
        t:Y
    })

classified = correct_prediction.eval(session=sess, feed_dict={x:X,t:Y})

prob = y.eval(session=sess, feed_dict={x:X})

print('classified:')
print(classified)
print()
print('output probability')
print(prob)




'''
###########################################################
Content : AND,OR,XOR gate - keras
###########################################################
'''
'''
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

np.random.seed(0)

model = Sequential([
    Dense(input_dim=2, output_dim=1),
    Activation('sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))

X=np.array([[0,0],[0,1],[1,0],[1,1]])
Y=np.array([[0],[1],[1],[1]])
#Y=np.array([[0],[0],[0],[1]])
#Y=np.array([[0],[1],[1],[0]])

model.fit(X,Y,epochs=200, batch_size=1)

classes = model.predict_classes(X,batch_size=1)
prob = model.predict_proba(X,batch_size=1)

print('classified:')
print(Y==classes)
print()
print('output probability')
print(prob)
'''


'''
###########################################################
Content : Multi-class
###########################################################
'''
'''
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

M=2
K=3
n=100
N=n*K
X1=np.random.randn(n,M)+np.array([0,10])
X2=np.random.randn(n,M)+np.array([5,5])
X3=np.random.randn(n,M)+np.array([10,0])
Y1=np.array([[1,0,0] for i in range(n)])
Y2=np.array([[0,1,0] for i in range(n)])
Y3=np.array([[0,0,1] for i in range(n)])

X=np.concatenate((X1,X2,X3),axis=0)
Y=np.concatenate((Y1,Y2,Y3),axis=0)

W=tf.Variable(tf.zeros([M,K]))
b=tf.Variable(tf.zeros([K]))

x=tf.placeholder(tf.float32, shape=[None, M])
t=tf.placeholder(tf.float32, shape=[None, K])
y=tf.nn.softmax(tf.matmul(x,W)+b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(t*tf.log(y),reduction_indices=[1]))

train_step=tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

correct_prediction = tf.equal(tf.arg_max(y,1),tf.argmax(t,1))

batch_size = 50
n_batches = N

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

for epoch in range(20):
    X_,Y_ = shuffle(X,Y)

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size

        sess.run(train_step, feed_dict={x:X_[start:end],
                                        t:Y_[start:end]})

X_,Y_=shuffle(X,Y)

classified = correct_prediction.eval(session=sess,feed_dict={
    x:X_[0:10],
    t:Y_[0:10]
})

prob = y.eval(session=sess, feed_dict={x:X_[0:10]})

print('classified')
print(classified)
print()
print('output probability:')
print(prob)
'''

'''
###########################################################
Content : Multi-class - keras
###########################################################
'''

'''
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD


import tensorflow as tf
from sklearn.utils import shuffle

M=2
K=3
n=100
N=n*K
X1=np.random.randn(n,M)+np.array([0,10])
X2=np.random.randn(n,M)+np.array([5,5])
X3=np.random.randn(n,M)+np.array([10,0])
Y1=np.array([[1,0,0] for i in range(n)])
Y2=np.array([[0,1,0] for i in range(n)])
Y3=np.array([[0,0,1] for i in range(n)])

X=np.concatenate((X1,X2,X3),axis=0)
Y=np.concatenate((Y1,Y2,Y3),axis=0)

W=tf.Variable(tf.zeros([M,K]))
b=tf.Variable(tf.zeros([K]))

x=tf.placeholder(tf.float32, shape=[None, M])
t=tf.placeholder(tf.float32, shape=[None, K])
y=tf.nn.softmax(tf.matmul(x,W)+b)

model=Sequential()
model.add(Dense(input_dim=M,units=K))
model.add(Activation('softmax'))

model.compile(optimizer=SGD(lr=0.1),loss='categorical_crossentropy')

minibatch_size = 50
model.fit(X,Y,epochs=20,batch_size=minibatch_size)

X_,Y_ = shuffle(X,Y)
classes = model.predict_classes(X_[0:10],batch_size=minibatch_size)
prob=model.predict_proba(X_[0:10],batch_size=minibatch_size)

print('classified')
print(np.argmax(model.predict(X_[0:10]),axis=1)==classes)
print()
print('output probability:')
print(prob)
'''