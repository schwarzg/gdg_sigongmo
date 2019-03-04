import tensorflow as tf
import numpy as np
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()

n_steps = 28
n_inputs = 28
n_neurons = 300
n_outputs = 10

learning_rate = 0.001
n_epochs=100
batch_size=150


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# X_train -> (60000,28,28)
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0 # (60000,784)
print(np.shape(X_train))
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0 # (10000, 784)
print(np.shape(X_test))

y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

X_valid, X_train = X_train[:5000], X_train[5000:] #인덱싱해서 validation set 만들기
y_valid, y_train = y_train[:5000], y_train[5000:] # 처음 5000개와 5000개 이후부터 나머지까지 인덱싱
X_test = X_test.reshape((-1, n_steps, n_inputs))
X_valid = X_valid.reshape((-1, n_steps, n_inputs))






X=tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y=tf.placeholder(tf.int32,[None])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

logits=tf.layers.dense(states, n_outputs)#활성화하기 전의 값
xentorpy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)

loss=tf.reduce_mean(xentorpy)
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op=optimizer.minimize(loss)

correct=tf.nn.in_top_k(logits,y,1) #자동으로 출력을 평가 (evaluation)
accuracy=tf.reduce_mean(tf.cast(correct, tf.float32)) #cast 함수는 텐서를 새로운 형태로 캐스팅

init=tf.global_variables_initializer()



def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print("epoch :", epoch, ", train acc :", acc_batch, ", val acc:", acc_valid)

