# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 18:50:52 2019

@author: MSPL
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

train = pd.read_csv("Titanic_data/train.csv")
test = pd.read_csv('Titanic_data/test.csv')

### Data 전처리
train= train.drop(['Cabin'],axis=1)
train= train.drop(['Ticket'],axis=1)
test= test.drop(['Cabin'],axis=1)
test= test.drop(['Ticket'],axis=1)

southampton= train[train["Embarked"]=="S"].shape[0]
cherbourg= train[train["Embarked"]=="C"].shape[0]
queenstown= train[train["Embarked"]=="Q"].shape[0]
train= train.fillna({"Embarked": "S"})
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked']= train['Embarked'].map(embarked_mapping)
test['Embarked']= test['Embarked'].map(embarked_mapping)

train_test_data = [train, test]
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand = False)

for dataset in train_test_data:
  dataset['Title']= dataset['Title'].replace(['Lady','Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona'],'Rare')
  
  dataset['Title']= dataset['Title'].replace(['Countess','Lady','Sir'],'Royal')
  dataset['Title']= dataset['Title'].replace('Mlle','Miss')
  dataset['Title']= dataset['Title'].replace('Ms','Miss')
  dataset['Title']= dataset['Title'].replace('Mme','Mrs')

title_mapping= {"Mr":1, "Miss":2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6,}
for dataset in train_test_data:
  dataset['Title']= dataset['Title'].map(title_mapping)
  dataset['Title']= dataset['Title'].fillna(0)

train= train.drop(['Name', 'PassengerId'],axis=1)
passengerid= test[["PassengerId"]].values
test= test.drop(['Name', 'PassengerId'],axis=1)
combine= [train, test]
sex_mapping= {"male":0, "female": 1}
for dataset in combine :
  dataset['Sex']=dataset['Sex'].map(sex_mapping)
  
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)

### Modeling!
train_data = train.drop(['Survived'],axis=1)
target= train['Survived']

M=8 # # of features
K=1 # # of classes 
#Hypothesis (modeling)
x=tf.placeholder(tf.float32, shape=[None,M])
y_target=tf.placeholder(tf.float32, shape=[None,K])

#layer1
n1= 12 # dimension
w1=tf.Variable(tf.zeros([M,n1]), name='weight1')
b1=tf.Variable(tf.zeros([n1]), name='bias1')
layer1=tf.nn.sigmoid(tf.matmul(x,w1)+b1)
#layer1 = tf.nn.dropout(layer1,keep_prob=0.5) # dropout

#layer2
w2=tf.Variable(tf.zeros([n1,K]), name='weight2')
b2=tf.Variable(tf.zeros([K]), name='bias2')
y=tf.nn.sigmoid(tf.matmul(layer1,w2)+b2)

#cost/ loss function
cost= -tf.reduce_sum(y_target*tf.log(y+1e-8)+(1-y_target)*tf.log(1-y+1e-8),axis=1)
#Minimize
optimizer= tf.train.AdamOptimizer(learning_rate=0.01)
train_model= optimizer.minimize(cost) 

# Test model
is_correct = tf.equal(tf.to_float(tf.greater(y, 0.5)), y_target)
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
test_result = tf.to_float(tf.greater(y, 0.5))

"모델 학습"
#Initializes global vaiables 
sess=tf.Session()
sess.run(tf.global_variables_initializer())

#Training
batch_size= 50
for epoch in range(2000):
    X_, Y_ =shuffle(train_data,target)
    X_=np.asarray(X_).astype('float32')
    Y_=np.asarray(Y_).astype('float32')
    Y_=np.expand_dims(Y_,axis=1)
    total_batch= int(891/batch_size)
    avg_cost=0
    
    for i in range(total_batch):
        batch_xs, batch_ys = X_[i*batch_size:i*batch_size+batch_size], Y_[i*batch_size:i*batch_size+batch_size]
        c,_= sess.run([cost, train_model], feed_dict = {x : batch_xs, y_target : batch_ys})
        avg_cost+= c/ total_batch
    
    if (epoch % 100) == 0: 
        train_accuracy = sess.run(accuracy,feed_dict = {x : X_, y_target : Y_})
        print('Epoch:', '%04d' % (epoch + 1), 'acc =',  train_accuracy)
                
target=np.expand_dims(target,axis=1)        
train_accuracy = sess.run(accuracy,feed_dict = {x : train_data, y_target : target})
print("\nTrain Acc: ", train_accuracy)
      
# Test the model using test sets
print (train[['Pclass', 'Fare']].groupby(['Pclass'], as_index=False).mean())
print("")
print(test[test["Fare"].isnull()]["Pclass"])
for dataset in combine:
  dataset['Fare'] = dataset['Fare'].fillna(13.675) # The only one empty fare data's pclass is 3.
test_data = test

test_accuracy = sess.run(test_result, feed_dict={x : test_data})
sub=np.concatenate((passengerid.astype(int), test_accuracy.astype(int)),axis=1)
sub=pd.DataFrame(sub,columns=['PassengerId','Survived'])
print(sub.head())
#sub.to_csv("submission_2_1.csv",index=False)
